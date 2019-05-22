package facec

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io/ioutil"
	"net/http"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"gopkg.in/mgo.v2/bson"

	"github.com/qiniu/xlog.v1"

	"qiniu.com/argus/argus/facec/client"
	"qiniu.com/argus/argus/facec/config"
	"qiniu.com/argus/argus/facec/db"
)

type ClusterWorkerConfig struct {
	URL        string        `json:"url"`
	Period     time.Duration `json:"period"`
	Concurrent int           `json:"concurrent"`
}

type ClusterWorker struct {
	config     *ClusterWorkerConfig
	groupMutex GroupMutex
	stopped    int32
	wg         sync.WaitGroup
}

func NewClusterWorker(cfg *ClusterWorkerConfig) *ClusterWorker {
	dao, _ := db.NewDataVersionDao()
	w := &ClusterWorker{
		config:     cfg,
		groupMutex: NewGroupMutex(dao),
	}
	w.run()
	return w
}

func (r *ClusterWorker) run() {
	for i := 0; i < r.config.Concurrent; i++ {
		r.wg.Add(1)
		go func() {
			defer func() {
				r.wg.Done()
			}()
			dao, _ := db.NewClusterTaskDao()
			for atomic.LoadInt32(&r.stopped) == 0 {
				ctx := context.Background()

				task, _ := dao.FindTask(ctx)
				if task == nil {
					time.Sleep(time.Second)
					continue
				}

				proc, _ := r.groupMutex.NewProcedure(ctx, task.UID, task.Euid, time.Second*600)

				buildCluster(ctx, proc, r.config.URL)

				_ = dao.DoneTask(ctx, *task)
			}
		}()
	}
}

func (r *ClusterWorker) Stop() {
	atomic.StoreInt32(&r.stopped, 1)
	r.wg.Wait()
}

func buildCluster(ctx context.Context, proc *GroupProcedure, clusterAPI string) error {

	var (
		xl     = xlog.FromContextSafe(ctx)
		uid, _ = strconv.ParseUint(proc.Uid, 10, 32)
		cli    = client.NewRPCClient(client.EvalEnv{Uid: uint32(uid)}, time.Second*3600)
	)

	faceDao, err := db.NewFaceDao()
	if err != nil {
		xl.Error("new face dao error", err)
		return err
	}
	faces, err := faceDao.FindByEuid(ctx, proc.Uid, proc.Euid)
	if err != nil {
		xl.Error("find face error", err)
		return err
	}
	faceCount := len(faces)
	if faceCount == 0 {
		xl.Info("no face to cluster. %s %s", proc.Uid, proc.Euid)
		return nil
	}

	type _ReqData struct {
		URI       string `json:"uri"`
		Attribute struct {
			ClusterID int64 `json:"cluster_id"`
			GtID      int64 `json:"gt_id"`
		} `json:"attribute"`
	}

	req := struct {
		Data []_ReqData `json:"data"`
	}{
		Data: make([]_ReqData, 0, faceCount),
	}
	for i := 0; i < faceCount; i++ {
		face := &faces[i]
		req.Data = append(req.Data,
			_ReqData{
				URI: _DataURIPrefix + face.Feature.Feature,
				Attribute: struct {
					ClusterID int64 `json:"cluster_id"`
					GtID      int64 `json:"gt_id"`
				}{
					ClusterID: int64(face.ClusterID),
					GtID:      int64(face.GtID),
				},
			})
	}

	xl.Info("feature count", len(req.Data), proc.Uid, proc.Euid)
	var resp = struct {
		Code    int64  `json:"code"`
		Message string `json:"message"`
		Result  struct {
			FacexCluster []struct {
				ClusterCenterDist float64 `json:"cluster_center_dist"`
				ClusterID         int64   `json:"cluster_id"`
			} `json:"facex_cluster"`
		} `json:"result"`
	}{}
	err = cli.CallWithJson(ctx, &resp, "POST", clusterAPI, req)
	xl.Debugf("post cluster err:%v, req: %d, url:%v", err, len(req.Data), clusterAPI)
	if err != nil {
		xl.Error("request cluster api error", proc.Uid, proc.Euid)
		return err // TODO
	}

	if resp.Code != 0 {
		xl.Errorf("request cluster api error code:%d,msg:%s", resp.Code, resp.Message)
		return fmt.Errorf(resp.Message)
	}

	xl.Infof("cluster result:%v", resp)
	if err = updateCluster(ctx, faceDao, faces, resp.Result.FacexCluster); err != nil {
		return err // TODO
	}

	if err = buildGroup2(ctx, proc, faces); err != nil {
		return err // TODO
	}

	return nil
}

var closeChan chan int

func updateCluster(ctx context.Context,
	faceDao db.FaceDao,
	faces []db.Face,
	f2c []struct {
		ClusterCenterDist float64 `json:"cluster_center_dist"`
		ClusterID         int64   `json:"cluster_id"`
	},
) error {
	xl := xlog.FromContextSafe(ctx)
	xl.Debug("update cluster")
	var face2Update []db.Face
	for i, s := range f2c {
		face := &faces[i]
		if face == nil {
			continue
		}

		var update bool
		if face.ClusterID != s.ClusterID {
			face.ClusterID = s.ClusterID
			update = true
		}

		if face.ClusterCenterDist != s.ClusterCenterDist {
			face.ClusterCenterDist = s.ClusterCenterDist
			update = true
		}

		if update {
			face2Update = append(face2Update, *face)
		}
	}

	if len(face2Update) == 0 {
		return nil
	}

	err := faceDao.UpdateClusterAndCenterDist(context.Background(), face2Update)
	if err != nil {
		xl.Warn("update cluster id error", err)
		return err
	}

	return nil
}

var locker = &sync.Mutex{}

func buildGroup2(ctx context.Context, proc *GroupProcedure, faces []db.Face) error {

	var (
		xl          = xlog.FromContextSafe(ctx)
		aliasDao, _ = db.NewAliasDao()
		now         = time.Now()

		groups []db.GroupInfo = make([]db.GroupInfo, 0)
		groupm map[int64]int  = make(map[int64]int)
	)

	xl.Info("build group", proc.Uid, proc.Euid)
	for i, face := range faces {
		xl.Infof("index:%v;face Id:%v,GtID:%v,clusterID:%v,dis:%v",
			i, face.ID, face.GtID, face.ClusterID, face.ClusterCenterDist)
	}
	for _, face := range faces {
		var (
			clusterID = calcGroupID(face.ClusterID, face.GtID)
			faceG     = db.FaceG{
				ID:                face.ID,
				File:              face.File,
				Pts:               face.Pts.Det,
				Score:             face.Score,
				ClusterCenterDist: face.ClusterCenterDist,
			}
		)
		if index, ok := groupm[clusterID]; ok {
			group := &groups[index]
			group.Group.FaceCount += 1
			group.Group.Faces = append(group.Group.Faces, faceG)
		} else {
			groupm[clusterID] = len(groups)
			groups = append(groups,
				db.GroupInfo{
					UID:  proc.Uid,
					Euid: proc.Euid,
					Group: db.Group{
						FaceCount: 1,
						ID:        clusterID,
						Version:   proc.GetVersion(),
						Faces:     []db.FaceG{faceG},
					},
					Modelv:    "v1", // TODO
					CreatedAt: now,
				},
			)
		}
	}

	for _, group := range groups {
		group.Group.Refs = calcCover2(group.Group.Faces)
	}
	aliasIDs, err := aliasDao.FindAlias(proc.Uid, proc.Euid)
	if err != nil {
		xl.Error("find all alias failed.", proc.Uid, proc.Euid, err)
		return err
	}

	var groups2 []db.GroupInfo = make([]db.GroupInfo, 0)
	groupm = make(map[int64]int)
	for _, group := range groups {
		clusterID := group.Group.ID
		if clusterID >= 0 && int64(len(aliasIDs)) > clusterID {
			clusterID = aliasIDs[clusterID]
		}
		if index, ok := groupm[clusterID]; ok {
			g := groups2[index]
			g.Group.FaceCount += group.Group.FaceCount
			g.Group.Faces = append(g.Group.Faces, group.Group.Faces...)
			if clusterID == group.Group.ID {
				g.Group.Refs = group.Group.Refs
			}
		} else {
			groupm[clusterID] = len(groups2)
			groups2 = append(groups2, group)
		}
	}

	groupDao, err := db.NewGroupDao()
	if err != nil {
		xl.Error("new group dao error")
		return err
	}
	xl.Infof("insert groups count %d\n", len(groups))
	if err = groupDao.Insert(ctx, groups); err != nil {
		xl.Error("insert group info error")
		return err
	}

	go func() {
		groupDao.RemoveByVersion(ctx, proc.Uid, proc.Euid, proc.GetOldVersion())
	}()

	return nil
}

func buildGroup(ctx context.Context,
	uid, euid string,
	file2FaceIDs map[string][]bson.ObjectId,
	faces []db.Face,
	clusterResp []struct {
		ClusterCenterDist float64 `json:"cluster_center_dist"`
		ClusterID         int64   `json:"cluster_id"`
	},
) error {

	xl := xlog.FromContextSafe(ctx)

	xl.Debug("build group", uid, euid)
	cluster2FaceIDs, err := buildCluster2FaceIDs(uid, euid, faces, clusterResp)
	if err != nil {
		xl.Error("build cluster of face error", err)
		return err
	}

	xl.Debugf("cluster2faceIDs:%v", cluster2FaceIDs)
	cluster2Files := buildCluster2Files(ctx, cluster2FaceIDs, file2FaceIDs)

	xl.Debugf("cluster2Files:%v", cluster2Files)
	var groups []db.GroupInfo
	now := time.Now()
	for clusterID, files := range cluster2Files {
		if len(files) == 0 {
			continue
		}
		if clusterID == UnknownClusterID {
			continue
		}
		groups = append(groups, db.GroupInfo{
			ID:        bson.NewObjectId(),
			Euid:      euid,
			UID:       uid,
			CreatedAt: now,
			Group: db.Group{
				FaceCount: int64(len(cluster2FaceIDs[clusterID])),
				Files:     files,
				ID:        int64(clusterID),
				Refs: calcCover(
					ctx,
					file2FaceIDs,
					cluster2FaceIDs[clusterID],
					faces),
			},
		})
	}

	groupDao, err := db.NewGroupDao()
	if err != nil {
		xl.Error("new group dao error")
		return err
	}

	oldIDs, err := groupDao.FindIDs(uid, euid)
	if err != nil {
		xl.Error("find group ids error", err)
		return err
	}
	xl.Debugf("old group id count %d\n", len(oldIDs))
	xl.Debugf("insert groups count %d\n", len(groups))
	err = groupDao.Insert(context.Background(), groups)
	if err != nil {
		xl.Error("insert group info error")
		return err
	}

	if len(oldIDs) == 0 {
		return nil
	}

	err = groupDao.RemoveAllByIDs(oldIDs...)
	if err != nil {
		xl.Error("remove the old group error")
	}
	return nil
}

func calcCover2(faces []db.FaceG) []db.Ref {

	var face *db.FaceG
	for _, fac := range faces {
		if face == nil || face.ClusterCenterDist > fac.ClusterCenterDist {
			face = &fac
		}
	}
	return []db.Ref{
		{
			File:  face.File,
			Pts:   face.Pts,
			Score: face.Score,
		},
	}
}

func calcCover(ctx context.Context,
	file2FaceIDs map[string][]bson.ObjectId,
	faceIDs []string,
	faces []db.Face) []db.Ref {

	var (
		xl = xlog.FromContextSafe(ctx)
	)
	var centerDist float64
	var face *db.Face

	for _, faceID := range faceIDs {
		f := findFace(bson.ObjectIdHex(faceID), faces)
		if f == nil {
			xl.Warnf("faceID:%s,object id:%s cannot find",
				faceID,
				bson.ObjectIdHex(faceID))
			continue
		}
		if face == nil || f.ClusterCenterDist < centerDist {
			face = f
			centerDist = f.ClusterCenterDist
			xl.Debugf("face center dist:%f", centerDist)
		}
	}

	if face == nil {
		xl.Errorf("cannot find cover:%v:%v", faces, faceIDs)
		return nil
	}

	var file string
	for f, ffaceIDs := range file2FaceIDs {
		for _, faceID := range ffaceIDs {
			if faceID == face.ID {
				file = f
				goto END
			}
		}
	}

END:
	xl.Debugf("found cover, file:%s, face:%f", file, face.ClusterCenterDist)
	return []db.Ref{
		{
			File:  file,
			Pts:   face.Pts.Det,
			Score: face.Score,
		},
	}
}

func buildCluster2Files(ctx context.Context,
	cluster2FaceIDs map[int][]string,
	file2FaceIDs map[string][]bson.ObjectId,
) map[int][]string {

	xl := xlog.FromContextSafe(ctx)

	cluster2Files := make(map[int][]string, len(cluster2FaceIDs))

	for file, fs := range file2FaceIDs {
		for _, id := range fs {
			clusterID := findClusterID(ctx, cluster2FaceIDs, id)
			xl.Debug("clusterID:", clusterID,
				",file count:", len(cluster2Files[clusterID]))
			cluster2Files[clusterID] = append(cluster2Files[clusterID], file)
		}
	}

	return cluster2Files
}

func findClusterID(ctx context.Context,
	cluster2FaceIDs map[int][]string,
	faceID bson.ObjectId) int {
	xl := xlog.FromContextSafe(ctx)
	for clusterID, faceIDs := range cluster2FaceIDs {
		for _, id := range faceIDs {
			if bson.ObjectIdHex(id) == faceID {
				return clusterID
			}
		}
	}

	xl.Warn("no cluster id of ", faceID)
	return UnknownClusterID
}

// the index of return value is the cluster id
func buildCluster2FaceIDs(
	uid, euid string,
	faces []db.Face,
	clusterResp []struct {
		ClusterCenterDist float64 `json:"cluster_center_dist"`
		ClusterID         int64   `json:"cluster_id"`
	},
) (map[int][]string, error) {
	aliasDao, err := db.NewAliasDao()
	if err != nil {
		return nil, err
	}

	aliasIDs, err := aliasDao.FindAlias(uid, euid)
	if err != nil {
		return nil, err
	}

	cluster2FaceIDs := make(map[int][]string)
	for i, c := range clusterResp {
		face := &faces[i]
		clusterID, faceID := c.ClusterID, face.ID.String()
		clusterID = calcGroupID(clusterID, face.GtID)
		if clusterID >= 0 && int64(len(aliasIDs)) > clusterID {
			clusterID = aliasIDs[clusterID]
		}

		cluster2FaceIDs[int(clusterID)] = append(cluster2FaceIDs[int(clusterID)], faceID)
	}

	return cluster2FaceIDs, nil
}

func ignoreNoFeatureFace(faces []db.Face) []db.Face {
	count := 0
	for _, f := range faces {
		if len(f.Feature.Feature) > 0 {
			count++
		}
	}

	retFaces := make([]db.Face, 0, count)
	for _, f := range faces {
		if len(f.Feature.Feature) > 0 {
			retFaces = append(retFaces, f)
		}
	}

	return retFaces
}

// Cluster cluster the faces
func Cluster(ctx context.Context, uid, euid, clusterAPI string) error {
	xl := xlog.FromContextSafe(ctx)
	xl.Debug("manual cluster")
	locker.Lock()
	defer locker.Unlock()

	return cluster(ctx, uid, euid, clusterAPI)
}

func cluster(ctx context.Context, uid, euid, clusterAPI string) error {
	xl := xlog.FromContextSafe(ctx)
	imageDao, err := db.NewImageDao()
	if err != nil {
		xl.Error("new image dao error")
		return err
	}

	xl.Debugf("cluster uid:%s,euid:%s", uid, euid)
	file2FaceIDs, err := imageDao.FindFaces(uid, euid)
	xl.Debugf("file2 faceIDs:%v", file2FaceIDs)
	if err != nil {
		return err
	}

	faceDao, err := db.NewFaceDao()
	if err != nil {
		xl.Error("new face dao error", err)
		return err
	}

	faceIDs := flatFaceIDs(file2FaceIDs)
	faces, err := faceDao.Find(faceIDs...)
	if err != nil {
		xl.Error("find face error", err)
		return err
	}

	faces = ignoreNoFeatureFace(faces)
	faceCount := len(faces)
	if faceCount == 0 {
		xl.Debug("no face to cluster")
		return nil
	}

	type _ReqData struct {
		URI       string `json:"uri"`
		Attribute struct {
			ClusterID int64 `json:"cluster_id"`
			GtID      int64 `json:"gt_id"`
		} `json:"attribute"`
	}

	req := struct {
		Data []_ReqData `json:"data"`
	}{
		Data: make([]_ReqData, 0, faceCount),
	}
	for i := 0; i < faceCount; i++ {
		face := &faces[i]
		req.Data = append(req.Data,
			_ReqData{
				URI: _DataURIPrefix + face.Feature.Feature,
				Attribute: struct {
					ClusterID int64 `json:"cluster_id"`
					GtID      int64 `json:"gt_id"`
				}{
					ClusterID: int64(face.ClusterID),
					GtID:      int64(face.GtID),
				},
			})
	}

	xl.Info("feature count", len(req.Data))
	var resp = struct {
		Code    int64  `json:"code"`
		Message string `json:"message"`
		Result  struct {
			FacexCluster []struct {
				ClusterCenterDist float64 `json:"cluster_center_dist"`
				ClusterID         int64   `json:"cluster_id"`
			} `json:"facex_cluster"`
		} `json:"result"`
	}{}
	var uintUID, _ = strconv.ParseUint(uid, 10, 64)
	cli := client.NewRPCClient(client.EvalEnv{Uid: uint32(uintUID)}, time.Second*600)
	err = cli.CallWithJson(context.Background(), &resp, "POST", clusterAPI, req)
	xl.Debugf("post cluster err:%v, %d", err, len(req.Data))
	if err != nil {
		xl.Error("request cluster api error")
		return err // TODO
	}

	if resp.Code != 0 {
		xl.Errorf("request cluster api error code:%d,msg:%s",
			resp.Code, resp.Message)
		return fmt.Errorf(resp.Message)
	}

	err = updateCluster(ctx, faceDao, faces, resp.Result.FacexCluster)
	if err != nil {
		return err // TODO
	}

	err = buildGroup(ctx, uid, euid, file2FaceIDs, faces, resp.Result.FacexCluster)
	if err != nil {
		return err // TODO
	}

	return nil
}

func equalOfPts(pts1 [][]int64, pts2 [][]int64) bool {
	if len(pts1) != len(pts2) {
		return false
	}

	for i, l := 0, len(pts1); i < l; i++ {
		p1, p2 := pts1[i], pts2[i]
		if len(p1) != len(p2) {
			return false
		}

		for i, len := 0, len(p1); i < len; i++ {
			if p1[i] != p2[i] {
				return false
			}
		}
	}

	return true
}

func findFace(faceID bson.ObjectId, faces []db.Face) *db.Face {
	for i, len := 0, len(faces); i < len; i++ {
		face := &faces[i]
		if face.ID == faceID {
			return face
		}
	}
	return nil
}

func url(api *config.Service) string {
	u, n := api.URL, api.Name
	if strings.HasSuffix(api.URL, "/") {
		return u + n
	}

	return u + "/" + n
}

// StopClusterWorker send the stop signal to the background job
func StopClusterWorker() {
	closeChan <- 0
}

func post(ctx context.Context, url string, params interface{}, ret interface{}, timeout int) error {
	xl := xlog.FromContextSafe(ctx)
	xl.Debugf("post %s", url)

	start := time.Now()
	defer func() {
		xl.Debugf("post %s, time cost: %f S",
			url,
			float64(time.Since(start))/float64(time.Second))
	}()

	data, _ := json.Marshal(params)
	client := &http.Client{
		Timeout: time.Duration(timeout) * time.Second,
	}
	resp, err := client.Post(url, "application/json", bytes.NewReader(data))

	if err != nil {
		xl.Warnf("request error:%v", err)
		return err
	}

	if resp.StatusCode != 200 {
		xl.Warnf("request error, resp:%v", resp)
		return fmt.Errorf("bad reponse code:%d", resp.StatusCode)
	}

	data, err = ioutil.ReadAll(resp.Body)
	resp.Body.Close()

	if err != nil {
		xl.Error("response error")
		return err
	}
	err = json.Unmarshal(data, ret)
	if err != nil {
		xl.Error("response data error")
	}
	return err
}
