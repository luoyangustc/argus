package facec

import (
	"context"
	"fmt"
	"strconv"
	"strings"
	"time"

	"gopkg.in/mgo.v2"
	"gopkg.in/mgo.v2/bson"

	"github.com/qiniu/xlog.v1"

	authstub "qiniu.com/auth/authstub.v1"
	"qiniu.com/argus/argus/facec/client"
	"qiniu.com/argus/argus/facec/db"
	"qiniu.com/argus/argus/facec/proto"
	"qiniu.com/argus/argus/monitor"
)

func (s *Service) PostFaceClusterGather(
	ctx context.Context,
	args *struct {
		Euid  string   `json:"euid"`
		Items []string `json:"items"`
	},
	env *authstub.Env,
) (resp *proto.PostImagesFacegroupsResp, err error) {

	xl, ok := xlog.FromContext(ctx)
	if !ok {
		ctx = xlog.NewContextWithRW(ctx, env.W, env.Req)
		xl, _ = xlog.FromContext(ctx)
	}
	//defer xl.Xtrack("ARGUS", begin, &err)
	defer func(begin time.Time) {
		xl.Xprof("ARGUS", begin, err)
		monitor.ResponseTime("PostImagesFacegather", err, time.Since(begin))
	}(time.Now())
	defer func() { xl.Debugf("resp:%#v err:%#v", resp, err) }()

	{
		if len(args.Items) > MAXImageNum {
			return nil, ErrImageNumExceeded
		}
		if args.Euid == "" {
			return nil, ErrEuidEmpty
		}
	}

	// imgs := s.getImgProcesser(args.Items)
	// imgs.FetchAll(ctx)

	xl.Debugf("PostImagesFacegather %#v", args)
	resp = &proto.PostImagesFacegroupsResp{}
	uid := uidToStr(env.Uid)

	// 调用人脸检测API
	// fDetectArgs := imgs.NewUrls()
	faceDex, err := s.cl.PostFacexDex(ctx, args.Items, client.EvalEnv{Uid: env.UserInfo.Uid, Utype: env.UserInfo.Utype})
	if err != nil {
		return nil, WrapWithLog(ctx, err, "cl.PostFacexDex error")
	}

	// 生成API返回结果
	{
		resp.Result.Fail = make([]proto.PostImagesFacegroupsResp_sub1, 0)
		resp.Modelv = MODEL_VERSION
		for i, img := range args.Items {
			if faceDex[i].Code != 0 {
				resp.Result.Fail = append(resp.Result.Fail, proto.PostImagesFacegroupsResp_sub1{
					Code:    faceDex[i].Code,
					Message: faceDex[i].Message,
					Item:    img,
				})
			}
		}
		// for _, v := range imgs.BadUrls() {
		// 	resp.Result.Fail = append(resp.Result.Fail, proto.PostImagesFacegroupsResp_sub1{
		// 		Item:    v.URL,
		// 		Code:    StatusCodeImageProcessError,
		// 		Message: v.Err,
		// 	})
		// }
	}

	// 保存图片和face到数据库
	var tasks []db.FeatureTask
	{
		begin2 := time.Now()
		var faces []db.Face

		for i, img := range args.Items {
			// 如果是检测失败的图片，跳过
			if faceDex[i].Code != 0 {
				continue
			}
			if len(faceDex[i].Result.Detections) == 0 {
				continue
			}
			// fileName := imgs.Revert(img).Url()
			for _, detection := range faceDex[i].Result.Detections {
				// if !CanDetectFace(imgs.Revert(img).RevertPts(detection.Pts)) {
				// 	xl.Debugf("%v cannot be detected", detection.Pts)
				// 	continue
				// }
				face := db.Face{
					ID:        bson.NewObjectId(),
					Score:     detection.Score,
					ClusterID: -2,
					GtID:      -1,
					CreatedAt: time.Now(),
					UID:       uid,
					Euid:      args.Euid,
					File:      img,
					Pts: db.FacePts{
						Det:    detection.Pts, // imgs.Revert(img).RevertPts(detection.Pts),
						ModelV: "v1",          // TODO
					},
				}
				task := db.FeatureTask{
					UID:       face.UID,
					Euid:      face.Euid,
					CreatedAt: face.CreatedAt,
				}
				task.Face.ID = face.ID
				task.Face.File = face.File
				task.Face.Pts = face.Pts
				faces = append(faces, face)
				tasks = append(tasks, task)
			}
			// 新逻辑怎么判断重复上传？？
		}

		if len(faces) > 0 {
			err = s.dFace.Insert(ctx, faces...)
			if err != nil {
				return nil, WrapWithLog(ctx, err, "dFace.Insert error")
			}
		}

		xl.Xprof("DB", begin2, nil)
		xl.Printf("save face. %d", time.Since(begin2)/time.Millisecond)
	}
	// 在表 聚类任务 中添加记录
	{
		if len(tasks) > 0 {
			// xl.Infof("%#v %#v %#v", s, s.dFeatureTask, tasks)
			err := s.dFeatureTask.Insert(ctx, tasks...)
			if err != nil {
				return nil, WrapWithLog(ctx, err, "erTask.Insert error")
			}
		}
	}
	resp.Message = "success"
	return resp, nil
}

type getImagesFacegroupsReq struct {
	Euid              string `json:"euid"`
	IncludeUncategory bool   `json:"include_uncategory"`
}

func (s *Service) GetFaceCluster(
	ctx context.Context,
	args *struct {
		Euid              string `json:"euid"`
		IncludeUncategory bool   `json:"include_uncategory"`
	},
	env *authstub.Env,
) (resp *proto.GetImagesFacegroupsResp, err error) {
	defer func(beginTime time.Time) {
		monitor.ResponseTime("GetImagesFacegroups", err, time.Since(beginTime))
	}(time.Now())
	{
		if args.Euid == "" {
			return nil, ErrEuidEmpty
		}
	}
	xl, ok := xlog.FromContext(ctx)
	if !ok {
		ctx = xlog.NewContextWithRW(ctx, env.W, env.Req)
		xl, _ = xlog.FromContext(ctx)
	}
	xl.Debugf("GetImagesFacegroups %#v", args)
	defer xl.Debugf("resp:%#v err:%#v", resp, err)
	resp = &proto.GetImagesFacegroupsResp{}
	uid := uidToStr(env.Uid)

	resp.Message = "success"
	resp.Modelv = MODEL_VERSION
	resp.Groups = make([]proto.GetImagesFacegroupsResp_sub2, 0)
	// 根据 euid uid 从表 分组信息 中查询并返回结果
	{
		var excludeGroupIDs []int
		if !args.IncludeUncategory {
			excludeGroupIDs = []int{-2, -1}
		}

		groups, err := s.dGroup.FindGroupWithExclude(uid, args.Euid, 0, 500, excludeGroupIDs...)
		if err != nil {
			return nil, WrapWithLog(ctx, err, "dGroup.FindGroups error")
		}
		for _, g := range groups {
			group := proto.GetImagesFacegroupsResp_sub2{
				ID:        g.Group.ID,
				FaceCount: g.Group.FaceCount,
			}
			for _, v := range g.Group.Faces {
				group.Faces = append(group.Faces,
					proto.GetImagesFacegroupsIDResp_sub1{
						URI:   v.File,
						Pts:   v.Pts,
						Score: v.Score,
					})
			}
			for _, v := range g.Group.Refs {
				group.Refs = append(group.Refs, proto.GetImagesFacegroupsResp_sub1{
					File:  v.File,
					Pts:   v.Pts,
					Score: v.Score,
				})
			}
			resp.Groups = append(resp.Groups, group)
		}
	}

	return resp, nil
}

type getImagesFacegroups_Req struct {
	Euid    string `json:"euid"`
	CmdArgs []string
}

func (s *Service) GetFaceCluster_(
	ctx context.Context,
	args *struct {
		CmdArgs []string
		Euid    string `json:"euid"`
	},
	env *authstub.Env,
) (resp *proto.GetImagesFacegroupsIDResp, err error) {
	xl, ok := xlog.FromContext(ctx)
	if !ok {
		ctx = xlog.NewContextWithRW(ctx, env.W, env.Req)
		xl, _ = xlog.FromContext(ctx)
	}
	defer func(beginTime time.Time) {
		monitor.ResponseTime("GetImagesFacegroups_", err, time.Since(beginTime))
	}(time.Now())
	xl.Debugf("GetImagesFacegroups_ %#v", args)
	defer xl.Debugf("resp:%#v err:%#v", resp, err)
	resp = &proto.GetImagesFacegroupsIDResp{}
	resp.Message = "success"
	resp.Modelv = MODEL_VERSION
	resp.Faces = make([]proto.GetImagesFacegroupsIDResp_sub1, 0)

	var finalID int64
	uid := uidToStr(env.Uid)

	// 参数检查
	{
		{
			if args.Euid == "" {
				return nil, ErrEuidEmpty
			}
		}
		g, err := strconv.Atoi(args.CmdArgs[0])
		if err != nil {
			return nil, ErrGroupNotExists
		}
		finalID = int64(g)
	}

	// 根据 euid uid 从表 分组信息 中查询所有图片标识
	// var files []string
	var faces []bson.ObjectId
	{
		group, err := s.dGroup.FindGroup(uid, args.Euid, finalID)
		if err == mgo.ErrNotFound {
			return nil, ErrGroupNotExists
		}
		if err != nil {
			return nil, WrapWithLog(ctx, err, "dGroup.FindGroup error")
		}
		// files = group.Group.Files
		for _, face := range group.Group.Faces {
			faces = append(faces, face.ID)
		}
	}
	// 根据 euid uid 和图片标识从表 图片信息中获取人脸特征id faces
	// var faces []bson.ObjectId
	// var images []db.Image
	// {
	// 	images, err = s.dImage.FindImageByFiles(uid, args.Euid, files)
	// 	if err != nil {
	// 		return nil, WrapWithLog(ctx, err, "dImage.FindImageByFiles error")
	// 	}
	// 	for _, v := range images {
	// 		faces = append(faces, v.Faces...)
	// 	}
	// }

	var faceDocs []db.Face
	{
		faceDocs, err = s.dFace.FindWithOutFeature(faces...)
		if err != nil {
			return nil, WrapWithLog(ctx, err, "dFace.Find error")
		}
	}

	// 根据 euid uid 从表 分组别名 查找 group_id
	var group []db.Alias
	{
		group, err = s.dAlias.FindGroup(uid, args.Euid, finalID)
		if err != nil {
			return nil, WrapWithLog(ctx, err, "dGroup.FindGroup error")
		}
	}

	isUsedFace := func(gID int64) bool {
		if finalID == gID {
			return true
		}
		for _, v := range group {
			if v.GroupID == gID {
				return true
			}
		}
		return false
	}

	// findImage := func(faceID bson.ObjectId) *db.Image {
	// 	// TODO 使用map
	// 	for _, v := range images {
	// 		for _, f := range v.Faces {
	// 			if f == faceID {
	// 				return &v
	// 			}
	// 		}
	// 	}
	// 	return nil
	// }

	// 根据 faces group_id 和请求分组id从 人脸特征 中查询特征
	{
		for _, f := range faceDocs {
			gID := calcGroupID(f.ClusterID, f.GtID)
			if !isUsedFace(gID) {
				continue
			}
			// 如果是需要返回的face
			face := proto.GetImagesFacegroupsIDResp_sub1{
				Pts:   f.Pts.Det,
				Score: f.Score,
				URI:   f.File,
			}
			// image := findImage(f.ID)
			// if image != nil {
			// 	face.File = image.File
			// }
			resp.Faces = append(resp.Faces, face)
		}
	}
	return resp, nil
}

func (s *Service) PostFaceClusterAdjust(
	ctx context.Context,
	args *struct {
		Euid        string   `json:"euid"`
		FromGroupID int64    `json:"from_group_id"`
		Items       []string `json:"items"`
		ToGroupID   int64    `json:"to_group_id"`
	},
	env *authstub.Env,
) (err error) {
	xl, ok := xlog.FromContext(ctx)
	if !ok {
		ctx = xlog.NewContextWithRW(ctx, env.W, env.Req)
		xl, _ = xlog.FromContext(ctx)
	}
	defer func(beginTime time.Time) {
		monitor.ResponseTime("PostImagesFacegroupsAdjust", err, time.Since(beginTime))
	}(time.Now())
	xl.Debugf("PostImagesFacegroupsAdjust %#v", args)

	//参数检查
	{
		if strings.TrimSpace(args.Euid) == "" || len(args.Items) == 0 || args.FromGroupID == args.ToGroupID {
			return ErrBadRequest("euid and two different candidate groups are required")
		}
	}
	uid := uidToStr(env.Uid)
	proc, err := s.groupMutex.NewProcedure(ctx, uid, args.Euid, time.Millisecond*100)
	if err != nil {
		xl.Errorf("failed to get db mutex lock:%v", err)
		return WrapWithLog(ctx, err, "version db is busy,please try again later")
	}
	oldVersion := proc.OldVersion //call version db to get oldversion

	{
		gps, err := s.dGroup.FindGroupByGroupID(uid, args.Euid, oldVersion, args.FromGroupID, args.ToGroupID)
		if err != nil {
			revErr := proc.Revert()
			xl.Errorf("faceadjust get group info error:%v,version db roll back error:%v", err, revErr)
			return ErrBadRequest("faceadjust get group info error")
		}
		if len(gps) < 2 {
			revErr := proc.Revert()
			xl.Errorf("faceadjust group id dones't exists:%v,version db roll back error:%v", gps, revErr)
			return ErrBadRequest("faceadjust group id dones't exist")
		}

		var fmgrp, togrp db.GroupInfo
		var fmvFaces, fleftFaces []db.FaceG
		var changeFmgRef bool
		var mvfaceIds []bson.ObjectId
		var newRef db.Ref

		fmgmp := make(map[string][]db.FaceG)
		togmp := make(map[string][]db.FaceG)
		newVersion := proc.Version
		createdAt := time.Now()

		if gps[0].Group.ID == args.FromGroupID {
			fmgrp = gps[0]
			togrp = gps[1]
		} else {
			fmgrp = gps[1]
			togrp = gps[0]
		}

		for _, f := range fmgrp.Group.Faces {
			fmgmp[f.File] = append(fmgmp[f.File], f)
		}
		for _, f := range togrp.Group.Faces {
			togmp[f.File] = append(togmp[f.File], f)
		}

		for _, t := range args.Items {
			if fmgmp[t] == nil {
				revErr := proc.Revert()
				xl.Errorf("faceadjust image dones't exists in the origin group:%v,version db roll back error:%v", t, revErr)
				return ErrBadRequest("faceadjust image dones't exists in the origin group")
			}
			if togmp[t] != nil {
				revErr := proc.Revert()
				xl.Errorf("faceadjust group id dones't exists:%v,version db roll back error:%v", gps, revErr)
				return ErrBadRequest("faceadjust group id dones't exist")
			}
			if t == fmgrp.Group.Refs[0].File {
				changeFmgRef = true
			}

			fmvFaces = append(fmvFaces, fmgmp[t]...)
			for _, f := range fmgmp[t] {
				mvfaceIds = append(mvfaceIds, f.ID)
			}
			delete(fmgmp, t)
		}

		if changeFmgRef {
			var clusterCenterDist = 2.0 //cluster_center_dist has a value between 0~1
			for _, f := range fmgmp {
				fleftFaces = append(fleftFaces, f...)
				for _, fa := range f {
					if fa.ClusterCenterDist < clusterCenterDist {
						newRef.File = fa.File
						newRef.Pts = fa.Pts
						newRef.Score = fa.Score
						clusterCenterDist = fa.ClusterCenterDist
					}
				}
			}
			fmgrp.Group.Refs[0] = newRef
		} else {
			for _, f := range fmgmp {
				fleftFaces = append(fleftFaces, f...)
			}
		}
		togrp.Group.Faces = append(fmgrp.Group.Faces, fmvFaces...)
		fmgrp.Group.Faces = fleftFaces
		togrp.Group.Version = newVersion
		togrp.Group.FaceCount = togrp.Group.FaceCount + int64(len(fmvFaces))
		togrp.CreatedAt = createdAt
		fmgrp.Group.Version = newVersion
		fmgrp.Group.FaceCount = int64(len(fleftFaces))
		fmgrp.CreatedAt = createdAt

		//set gt_id
		err = s.dFace.UpdateGtId(ctx, args.ToGroupID, mvfaceIds...)
		if err != nil {
			revErr := proc.Revert()
			xl.Errorf("update gt_id error:%v,version db roll back error:%v", err, revErr)
			return WrapWithLog(ctx, err, "update gt_id in face table error")
		}
		//update version of all groups except from group
		err = s.dGroup.UpdateVersion(uid, args.Euid, newVersion, fmgrp.Group.ID)
		if err != nil {
			revErr := proc.Revert()
			xl.Errorf("update version error:%v,version db roll back error:%v", err, revErr)
			return WrapWithLog(ctx, err, "update group version  error")
		}

		if len(fmgrp.Group.Faces) == 0 {
			//remove from group
			err = s.dGroup.Insert(context.Background(), []db.GroupInfo{togrp})
		} else {
			err = s.dGroup.Insert(context.Background(), []db.GroupInfo{togrp, fmgrp})
		}
		if err != nil {
			revErr := proc.Revert()
			xl.Errorf("insert groups error:%v,version db roll back error:%v", err, revErr)
			return WrapWithLog(ctx, err, "insert groups error")
		}

		//有一个问题，如果在update group.version之后更新 data version 表失败了，则旧版本的group都不见了。。。。
		err = proc.Commit()
		if err != nil {
			xl.Errorf("udpate version error:%v", err)
			return WrapWithLog(ctx, err, "udpate version error")
		}
		err = s.dGroup.RemoveGroupsWithVersion(uid, args.Euid, oldVersion, fmgrp.Group.ID, fmgrp.Group.ID)
		if err != nil {
			xl.Errorf("udpate remove groups with old version:%v  error:%v", oldVersion, err)
			return WrapWithLog(ctx, err, "update remove groups with old version error")
		}
	}
	return nil
}

func (s *Service) PostFaceClusterMerge(
	ctx context.Context,
	args *struct {
		Euid      string  `json:"euid"`
		Groups    []int64 `json:"groups"`
		ToGroupID int64   `json:"to_group_id"`
	},
	env *authstub.Env,
) (err error) {
	xl, ok := xlog.FromContext(ctx)
	if !ok {
		ctx = xlog.NewContextWithRW(ctx, env.W, env.Req)
		xl, _ = xlog.FromContext(ctx)
	}
	defer func(beginTime time.Time) {
		monitor.ResponseTime("PostImagesFacegroupsMerge", err, time.Since(beginTime))
	}(time.Now())
	xl.Debugf("PostImagesFacegroupsMerge %#v", args)

	var groups []int64
	var faces []db.FaceG
	var togrp db.GroupInfo
	//参数检查
	{
		for _, g := range args.Groups {
			if g == args.ToGroupID {
				return ErrBadRequest("invalid group id in candiate groups")
			}
		}
		if strings.TrimSpace(args.Euid) == "" || len(args.Groups) == 0 {
			return ErrBadRequest("euid and candidate groups are required")
		}
		xl.Debugf("args.groups: %#v", args.Groups)
	}

	uid := uidToStr(env.Uid)
	proc, err := s.groupMutex.NewProcedure(ctx, uid, args.Euid, time.Millisecond*100)
	if err != nil {
		xl.Errorf("failed to get db mutex lock:%v", err)
		return WrapWithLog(ctx, err, "version db is busy,please try again later")
	}
	groups = append(groups, args.Groups...)
	groups = append(groups, args.ToGroupID)
	newVersion := proc.Version
	createdAt := time.Now()
	oldVersion := proc.OldVersion

	{
		grps, err := s.dGroup.FindGroupByGroupID(uid, args.Euid, oldVersion, groups...)
		if err != nil {
			xl.Errorf("get groups info error:%v", err)
			return WrapWithLog(ctx, err, "get groups info  error")
		}
		if len(grps) != len(args.Groups)+1 {
			xl.Errorf("invalid or duplicated group id provided:%v", grps)
			return WrapWithLog(ctx, err, "invalid or duplicated group id provided")
		}
		for _, g := range grps {
			if g.Group.ID == args.ToGroupID {
				togrp = g
			}
			faces = append(faces, g.Group.Faces...)
		}
	}

	{
		//更新group表
		err = s.dGroup.UpdateVersion(uid, args.Euid, newVersion, args.Groups...)
		if err != nil {
			xl.Errorf("update group version error:%v", err)
			return WrapWithLog(ctx, err, "update group version error")
		}
		togrp.Group.Faces = faces
		togrp.Group.FaceCount = int64(len(faces))
		togrp.CreatedAt = createdAt
		togrp.Group.Version = newVersion
		err = s.dGroup.Insert(context.Background(), []db.GroupInfo{togrp})
		if err != nil {
			xl.Errorf("insert latest group error:%v", err)
			return WrapWithLog(ctx, err, "insert latest group error")
		}

		//更新alias表
		err = s.dAlias.UpdateAlias(uid, args.Euid, args.ToGroupID, args.Groups...)
		if err != nil {
			return WrapWithLog(ctx, err, "update alias error")
		}

	}

	{
		//有一个问题，如果在update version之后更新 data version 表失败了，则旧版本的group都不见了。。。。
		err = proc.Commit()
		if err != nil {
			xl.Errorf("udpate version error:%v", err)
			return WrapWithLog(ctx, err, "udpate version error")
		}
	}

	{
		err = s.dGroup.RemoveGroupsWithVersion(uid, args.Euid, oldVersion, groups...)
		if err != nil {
			xl.Errorf("remove groups with old version:%v  error:%v", oldVersion, err)
			return WrapWithLog(ctx, err, "update remove groups with old version error")
		}
	}
	return
}

type postImagesFacegroupsRestartReq struct {
	Euid string `json:"euid"`
}

func (s *Service) PostImagesFacegroupsRestart(
	ctx context.Context,
	args *postImagesFacegroupsRestartReq,
	env *authstub.Env,
) (err error) {
	defer func(beginTime time.Time) {
		monitor.ResponseTime("PostImagesFacegroupsRestart", err, time.Since(beginTime))
	}(time.Now())
	xl := xlog.New(env.W, env.Req)
	xl.Debugf("PostImagesFacegroupsRestart %#v", args)
	if args.Euid == "" {
		return ErrEuidEmpty
	}
	go Cluster(ctx, uidToStr(env.Uid), args.Euid, s.cfg.Hosts.FacexCluster+"/v1/eval/facex-cluster")
	return
}

func uidToStr(uid uint32) string {
	return fmt.Sprintf("%d", uid)
}
