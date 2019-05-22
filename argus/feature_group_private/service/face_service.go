package service

import (
	"context"
	"fmt"
	"net/http"
	"sort"
	"sync"

	"github.com/pkg/errors"
	"github.com/qiniu/http/httputil.v1"
	"github.com/qiniu/http/restrpc.v1"
	"github.com/qiniu/uuid"
	xlog "github.com/qiniu/xlog.v1"

	"qiniu.com/argus/com/util"
	"qiniu.com/argus/feature_group_private"
	"qiniu.com/argus/feature_group_private/proto"
)

var _ feature_group.IFaceGroupService = new(FaceService)

type FaceService struct {
	Config FaceGroupsConfig
	groups feature_group.IFaceGroups
}

func NewFaceService(ctx context.Context, baseGroups *BaseGroups, config FaceGroupsConfig) (feature_group.IFaceGroupService, error) {
	xl := xlog.FromContextSafe(ctx)
	groups, err := NewFaceGroups(ctx, baseGroups, config)
	if err != nil {
		xl.Error("NewBaseGroups failed", err)
		return nil, err
	}
	s := &FaceService{
		groups: groups,
		Config: config,
	}
	return s, err
}

func (s *FaceService) initContext(ctx context.Context, env *restrpc.Env) (*xlog.Logger, context.Context) {
	xl, ok := xlog.FromContext(ctx)
	if !ok {
		xl = xlog.New(env.W, env.Req)
		ctx = xlog.NewContext(ctx, xl)
	}
	return xl, ctx
}

func (s *FaceService) PostGroups_Add(ctx context.Context,
	args *struct {
		CmdArgs []string
		Image   proto.Image `json:"image"`
		Params  struct {
			RejectBadFace bool `json:"reject_bad_face"`
		} `json:"params,omitempty"`
	},
	env *restrpc.Env,
) (*struct {
	ID          proto.FeatureID   `json:"id"`
	BoundingBox proto.BoundingBox `json:"bounding_box"`
	FaceQuality proto.FaceQuality `json:"face_quality,omitempty"`
	FaceNumber  int               `json:"face_number,omitempty"`
}, error) {
	xl, ctx := s.initContext(ctx, env)

	groupName := args.CmdArgs[0]
	if len(groupName) == 0 {
		return nil, httputil.NewError(http.StatusBadRequest, "invalid arguments")
	}
	if args.Image.ID == "" {
		s, err := uuid.Gen(16)
		if err != nil {
			return nil, httputil.NewError(http.StatusInternalServerError, "failed to gen uuid")
		}
		args.Image.ID = proto.FeatureID(s)
	}

	if args.Image.Tag == "" {
		args.Image.Tag = feature_group.DEFAULT_FEATURE_TAG
	}

	group, err := s.groups.Get(ctx, proto.GroupName(groupName))
	if err != nil {
		xl.Errorf("PostGroups_Add groups.Get group %s failed, error: %s", groupName, err)
		return nil, httputil.NewError(http.StatusBadRequest, "group not exist")
	}
	// add one image once
	// TODO: add serval images once
	boxes, faceNum, err := group.AddFace(ctx, args.Params.RejectBadFace, args.Image)
	if err != nil {
		xl.Error("PostGroups_Add group.AddFace failed:", err)
		switch err {
		case ErrNoFaceFound:
			fallthrough
		case ErrMultiFaceFound:
			return nil, httputil.NewError(http.StatusBadRequest, errors.Wrap(err, fmt.Sprintf("face number %d", faceNum)).Error())
		case ErrBlurFace, ErrSmallFace, ErrCoverFace, ErrPoseFace, ErrOrientationNotUp:
			return nil, httputil.NewError(http.StatusBadRequest, err.Error())
		case ErrParseFeature:
			return nil, httputil.NewError(http.StatusInternalServerError, err.Error())
		default:
			return nil, httputil.NewError(http.StatusInternalServerError, "failed to add face")
		}
	}

	xl.Debugf("Group %s add faces %s, boundingbox %v, quality %v", groupName, args.Image.ID, boxes[0].BoundingBox, boxes[0].Quality)

	// will not return quality_score back to user, https://jira.qiniu.io/browse/ATLAB-7610

	ret := &struct {
		ID          proto.FeatureID   `json:"id"`
		BoundingBox proto.BoundingBox `json:"bounding_box"`
		FaceQuality proto.FaceQuality `json:"face_quality,omitempty"`
		FaceNumber  int               `json:"face_number,omitempty"`
	}{ID: args.Image.ID, BoundingBox: boxes[0].BoundingBox, FaceQuality: boxes[0].Quality}
	if faceNum > 1 {
		ret.FaceNumber = faceNum
	}

	return ret, nil
}

func (s *FaceService) PostGroups_Update(ctx context.Context,
	args *struct {
		CmdArgs []string
		Image   proto.Image `json:"image"`
		Params  struct {
			RejectBadFace bool `json:"reject_bad_face"`
		} `json:"params,omitempty"`
	},
	env *restrpc.Env,
) error {
	xl, ctx := s.initContext(ctx, env)

	groupName := args.CmdArgs[0]
	if len(groupName) == 0 {
		return httputil.NewError(http.StatusBadRequest, "invalid arguments")
	}

	if args.Image.Tag == "" {
		args.Image.Tag = feature_group.DEFAULT_FEATURE_TAG
	}

	group, err := s.groups.Get(ctx, proto.GroupName(groupName))
	if err != nil {
		xl.Error("PostGroups_Update groups.Get failed:", err)
		return httputil.NewError(http.StatusBadRequest, "group not exist")
	}
	err = group.UpdateFace(ctx, args.Params.RejectBadFace, args.Image)
	if err != nil {
		xl.Error("PostGroups_Update group.UpdateFace failed:", err)
		return httputil.NewError(http.StatusInternalServerError, "failed to update face")
	}

	xl.Debugf("Group %s update faces", groupName)
	return nil
}

func (s *FaceService) PostGroups_Search(ctx context.Context,
	args *struct {
		CmdArgs    []string
		Images     []proto.ImageURI `json:"images"`
		Threshold  float32          `json:"threshold"`
		Limit      int              `json:"limit"`
		UseQuality bool             `json:"use_quality"`
	},
	env *restrpc.Env,
) (feature_group.FaceSearchResp, error) {

	xl, ctx := s.initContext(ctx, env)

	data := make([]proto.Data, len(args.Images))
	for i, pic := range args.Images {
		data[i].URI = pic
	}
	paramStruct := &struct {
		Images       []proto.Data
		Groups       []proto.GroupName
		ClusterGroup proto.GroupName
		Threshold    float32
		Limit        int
		UseQuality   bool
	}{
		Images:     data,
		Groups:     []proto.GroupName{proto.GroupName(args.CmdArgs[0])},
		Threshold:  args.Threshold,
		Limit:      args.Limit,
		UseQuality: args.UseQuality,
	}
	resp, err := s.faceSearch(ctx, paramStruct)
	if err != nil {
		xl.Errorf("call PostGroups_Search failed. %v", err)
		return resp, httputil.NewError(http.StatusInternalServerError, "failed to search face")
	}

	for i, obj := range resp.SearchResults {
		for j, outface := range obj.Faces {
			for k := range outface.Faces {
				resp.SearchResults[i].Faces[j].Faces[k].Group = ""
			}
		}
	}

	return resp, nil
}

func (s *FaceService) PostGroupsMultiSearch(ctx context.Context,
	args *struct {
		Images       []proto.ImageURI  `json:"images"`
		Groups       []proto.GroupName `json:"groups"`
		ClusterGroup proto.GroupName   `json:"cluster_group"`
		Threshold    float32           `json:"threshold"`
		Limit        int               `json:"limit"`
		UseQuality   bool              `json:"use_quality"`
	},
	env *restrpc.Env,
) (feature_group.FaceSearchResp, error) {
	xl, ctx := s.initContext(ctx, env)
	data := make([]proto.Data, len(args.Images))
	for i, pic := range args.Images {
		data[i].URI = pic
	}
	paramStruct := &struct {
		Images       []proto.Data
		Groups       []proto.GroupName
		ClusterGroup proto.GroupName
		Threshold    float32
		Limit        int
		UseQuality   bool
	}{
		Images:       data,
		Groups:       args.Groups,
		ClusterGroup: args.ClusterGroup,
		Threshold:    args.Threshold,
		Limit:        args.Limit,
		UseQuality:   args.UseQuality,
	}
	resp, err := s.faceSearch(ctx, paramStruct)
	if err != nil {
		xl.Errorf("call PostGroupsMultiSearch failed. %v", err)
		return resp, httputil.NewError(http.StatusInternalServerError, "failed to search face")
	}
	return resp, nil
}

func (s *FaceService) GetGroups_Images(ctx context.Context,
	args *struct {
		CmdArgs []string
		Tag     proto.FeatureTag `json:"tag"`
		Marker  string           `json:"marker"`
		Limit   int              `json:"limit"`
	},
	env *restrpc.Env,
) (feature_group.ImageListResp, error) {
	xl, ctx := s.initContext(ctx, env)
	resp := feature_group.ImageListResp{}

	groupName := args.CmdArgs[0]
	if len(groupName) == 0 {
		return resp, httputil.NewError(http.StatusBadRequest, "invalid arguments")
	}
	if 0 >= args.Limit {
		args.Limit = feature_group.DEFAULT_LIST_LIMIT
	}

	group, err := s.groups.Get(ctx, proto.GroupName(groupName))
	if err != nil {
		xl.Error("GetGroups_Images groups.Get failed:", err)
		return resp, httputil.NewError(http.StatusBadRequest, "group not exist")
	}

	resp.Images, resp.Marker, err = group.ListImage(ctx, args.Tag, args.Marker, args.Limit)
	if err != nil {
		xl.Error("GetGroups_Images group.ListImages failed:", err)
		return resp, httputil.NewError(http.StatusInternalServerError, "failed to list face-images")
	}

	xl.Debugf("Group %s list images, tag %s marker %s, limit %d", groupName, args.Tag, args.Marker, args.Limit)
	return resp, nil
}

func (s *FaceService) PostGroups_PtsSearch(ctx context.Context,
	args *struct {
		CmdArgs []string
		Data    []proto.Data `json:"data"`
		Params  struct {
			Threshold  float32 `json:"threshold"`
			Limit      int     `json:"limit"`
			UseQuality bool    `json:"use_quality,omitempty"`
		} `json:"params,omitempty"`
	}, env *restrpc.Env,
) (feature_group.FaceSearchResp, error) {
	xl, ctx := s.initContext(ctx, env)

	paramStruct := &struct {
		Images       []proto.Data
		Groups       []proto.GroupName
		ClusterGroup proto.GroupName
		Threshold    float32
		Limit        int
		UseQuality   bool
	}{
		Images:     args.Data,
		Groups:     []proto.GroupName{proto.GroupName(args.CmdArgs[0])},
		Threshold:  args.Params.Threshold,
		Limit:      args.Params.Limit,
		UseQuality: false,
	}
	resp, err := s.faceSearch(ctx, paramStruct)
	if err != nil {
		xl.Errorf("call PostGroups_PtsSearch failed. %v", err)
		return resp, httputil.NewError(http.StatusInternalServerError, "failed to search face")
	}

	for i, obj := range resp.SearchResults {
		for j, outface := range obj.Faces {
			for k := range outface.Faces {
				resp.SearchResults[i].Faces[j].Faces[k].Group = ""
			}
		}
	}

	return resp, nil
}

func (s *FaceService) PostGroupsMultiPtsSearch(ctx context.Context,
	args *struct {
		Data   []proto.Data `json:"data"`
		Params struct {
			Threshold  float32 `json:"threshold"`
			Limit      int     `json:"limit"`
			UseQuality bool    `json:"use_quality,omitempty"`
		} `json:"params,omitempty"`
		Groups       []proto.GroupName `json:"groups"`
		ClusterGroup proto.GroupName   `json:"cluster_group"`
	},
	env *restrpc.Env,
) (feature_group.FaceSearchResp, error) {
	xl, ctx := s.initContext(ctx, env)
	paramStruct := &struct {
		Images       []proto.Data
		Groups       []proto.GroupName
		ClusterGroup proto.GroupName
		Threshold    float32
		Limit        int
		UseQuality   bool
	}{
		Images:       args.Data,
		Groups:       args.Groups,
		ClusterGroup: args.ClusterGroup,
		Threshold:    args.Params.Threshold,
		Limit:        args.Params.Limit,
		UseQuality:   false,
	}
	resp, err := s.faceSearch(ctx, paramStruct)
	if err != nil {
		xl.Errorf("call PostGroupsMultiPtsSearch failed. %v", err)
		return resp, httputil.NewError(http.StatusInternalServerError, "failed to search face")
	}
	return resp, nil
}

func (s *FaceService) faceSearch(ctx context.Context,
	args *struct {
		Images       []proto.Data
		Groups       []proto.GroupName
		ClusterGroup proto.GroupName
		Threshold    float32
		Limit        int
		UseQuality   bool
	},
) (feature_group.FaceSearchResp, error) {
	xl := xlog.FromContextSafe(ctx)

	if args.Limit == 0 {
		args.Limit = defaultSearchLimit
	}
	resp := feature_group.FaceSearchResp{}

	groupIds := make([]proto.GroupName, 0)
	if len(args.Groups) != 0 {
		groupIds = removeSameGroupsName(args.Groups)
		if len(groupIds) > s.Config.GroupsNumber {
			return resp, httputil.NewError(http.StatusBadRequest, "too many groups")
		}
		if len(groupIds) == 0 {
			return resp, httputil.NewError(http.StatusBadRequest, "invalid arguments")
		}
	}
	if len(args.ClusterGroup) != 0 {
		groupIds = append(groupIds, args.ClusterGroup)
	}

	if args.Threshold < 0 || args.Threshold > 1 {
		return resp, httputil.NewError(http.StatusBadRequest, "invalid threshold")
	}
	if 0 >= args.Limit {
		args.Limit = feature_group.DEFAULT_LIST_LIMIT
	}

	ifg := make([]feature_group.IFaceGroup, 0) //组数
	for _, obj := range groupIds {
		group, err := s.groups.Get(ctx, obj)
		if err != nil {
			xl.Error("PostGroups_Search groups.Get failed:", err)
			return resp, httputil.NewError(http.StatusBadRequest, "group not exist")
		}
		ifg = append(ifg, group)
	}

	//检测和提特征(fd,ff)
	fvs, faceBoxes, err := s.groups.DetectAndFetchFeature(ctx, args.UseQuality, args.Images)
	if err != nil {
		xl.Errorf("fail to gain features : %s", err)
		return resp, err
	}

	var lock sync.Mutex
	var goerr error

	res := make([][]feature_group.FaceSearchRespItem, len(fvs))
	for i := range res {
		res[i] = make([]feature_group.FaceSearchRespItem, len(fvs[i]))
	}

	wg := sync.WaitGroup{}
	wg.Add(len(ifg))
	for _, gn := range ifg {
		go func(ctx context.Context, group feature_group.IFaceGroup) {
			//特征搜索(fs)
			defer wg.Done()
			results, err := group.SearchFace(ctx, args.Threshold, args.Limit, fvs) //results没有外层BoundingBox
			if err != nil {
				xlog.FromContextSafe(ctx).Error("PostGroups_Search group.SearchFace failed:", err)
				goerr = err
				return
			}
			lock.Lock()
			for i, pic := range results { //pic 一张图
				for j, face := range pic { //face 一张脸
					res[i][j].Faces = append(res[i][j].Faces, face.Faces...)
				}
			}
			lock.Unlock()
		}(util.SpawnContext(ctx), gn)
	}
	wg.Wait()
	if goerr != nil {
		return resp, goerr
	}

	//排序
	sortresult := make([][]feature_group.FaceSearchRespItem, len(fvs))
	for i, _ := range sortresult {
		sortresult[i] = make([]feature_group.FaceSearchRespItem, len(fvs[i]))
	}

	var comerr error
	var clustererr error
	var clusterGroup feature_group.IFaceGroup
	if len(args.ClusterGroup) != 0 {
		clusterGroup, clustererr = s.groups.Get(ctx, args.ClusterGroup)
		if clustererr != nil {
			xl.Errorf("ClusterSearch.Get group %s failed, error: %s", args.ClusterGroup, clustererr)
			return resp, clustererr
		}
	}

	addFeatureSlice := make([]proto.Feature, 0)
	for i, pic := range res {
		for j := range pic {
			sort.Slice(pic[j].Faces, func(k, l int) bool { return pic[j].Faces[k].Score > pic[j].Faces[l].Score })
			sortresult[i][j].BoundingBox = faceBoxes[i][j]
			if len(pic[j].Faces) > args.Limit {
				pic[j].Faces = pic[j].Faces[:args.Limit]
			}
			sortresult[i][j].Faces = append(sortresult[i][j].Faces, pic[j].Faces...)

			//人脸聚类排重
			if len(args.ClusterGroup) != 0 && len(sortresult[i][j].Faces) == 0 {
				//匹配中，按原来的逻辑
				//没有匹配中，则入库（聚类库）score为1
				var features proto.Feature
				str, err := uuid.Gen(16)
				if err != nil {
					comerr = err
					break
				}
				features.ID = proto.FeatureID(str)
				features.Value = fvs[i][j]
				features.Group = args.ClusterGroup
				features.BoundingBox = faceBoxes[i][j]

				addFeatureSlice = append(addFeatureSlice, features)

				//TODO 不需要搜了（已知结果）
				sortresult[i][j].Faces = make([]feature_group.FaceSearchRespFaceItem, 1)
				sortresult[i][j].Faces[0].ID = features.ID
				sortresult[i][j].Faces[0].Score = 1
				sortresult[i][j].Faces[0].Group = features.Group
				sortresult[i][j].Faces[0].BoundingBox = faceBoxes[i][j]
			}
		}
		if comerr != nil {
			break
		}
	}
	if comerr != nil {
		return resp, comerr
	}
	if len(args.ClusterGroup) != 0 && len(addFeatureSlice) != 0 {
		adderr := clusterGroup.AddFeature(ctx, addFeatureSlice...)
		if adderr != nil {
			return resp, adderr
		}
	}

	for _, row := range sortresult {
		faceSearchRespResults := feature_group.FaceSearchRespResults{}
		faceSearchRespResults.Faces = append(faceSearchRespResults.Faces, row...)
		if len(row) == 0 {
			faceSearchRespResults.Faces = make([]feature_group.FaceSearchRespItem, 0)
		}
		resp.SearchResults = append(resp.SearchResults, faceSearchRespResults)
	}
	if len(sortresult) == 0 {
		faceSearchRespResults := feature_group.FaceSearchRespResults{}
		faceSearchRespResults.Faces = make([]feature_group.FaceSearchRespItem, 0)
		resp.SearchResults = append(resp.SearchResults, faceSearchRespResults)
	}

	return resp, nil
}

func removeSameGroupsName(groupsArray []proto.GroupName) []proto.GroupName {
	maps := make(map[proto.GroupName]struct{})
	res := []proto.GroupName{}
	for _, obj := range groupsArray {
		if _, v := maps[obj]; !v {
			maps[obj] = struct{}{}
			res = append(res, obj)
		}
	}
	return res
}

func (s *FaceService) PostGroups_Cluster(ctx context.Context,
	args *struct {
		CmdArgs    []string
		Images     []proto.ImageURI `json:"images"`
		Threshold  float32          `json:"threshold"`
		Limit      int              `json:"limit"`
		UseQuality bool             `json:"use_quality"`
	},
	env *restrpc.Env,
) (feature_group.FaceSearchResp, error) {
	xl, ctx := s.initContext(ctx, env)
	data := make([]proto.Data, len(args.Images))
	for i, pic := range args.Images {
		data[i].URI = pic
	}
	paramStruct := &struct {
		Images       []proto.Data
		Groups       []proto.GroupName
		ClusterGroup proto.GroupName
		Threshold    float32
		Limit        int
		UseQuality   bool
	}{
		Images:       data,
		ClusterGroup: proto.GroupName(args.CmdArgs[0]),
		Threshold:    args.Threshold,
		Limit:        args.Limit,
		UseQuality:   args.UseQuality,
	}
	resp, err := s.faceSearch(ctx, paramStruct)
	if err != nil {
		xl.Errorf("call PostGroups_Cluster failed. %v", err)
		return resp, httputil.NewError(http.StatusInternalServerError, "failed to search face")
	}
	return resp, nil

}
