package manager

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"io"
	"io/ioutil"
	"net/http"
	"sort"
	"strings"
	"sync"
	"time"

	httputil "github.com/qiniu/http/httputil.v1"
	"github.com/qiniu/http/restrpc.v1"
	"github.com/qiniu/xlog.v1"
	"qiniu.com/argus/AIProjects/tianyan/serving"
)

const (
	defaultFaceLimit     = 5
	defaultMinFaceWidth  = 50
	defaultMinFaceHeight = 50
)

type BaseReq struct {
	CmdArgs []string
	ReqBody io.ReadCloser
}

func (s *Service) initContext(ctx context.Context, env *restrpc.Env) (*xlog.Logger, context.Context) {
	xl, ok := xlog.FromContext(ctx)
	if !ok {
		xl = xlog.New(env.W, env.Req)
		ctx = xlog.NewContext(ctx, xl)
	}
	return xl, ctx
}

func (s *Service) filterSmallFace(faces []serving.EvalFaceDetection) (ret []serving.EvalFaceDetection) {
	// 忽略所有50x50已下的人脸

	for _, detect := range faces {
		width := detect.Pts[1][0] - detect.Pts[0][0]
		height := detect.Pts[2][1] - detect.Pts[1][1]

		if width >= s.MinFaceWidth && height >= s.MinFaceHeight {
			ret = append(ret, detect)
		}
	}
	return
}

func (s *Service) parseFace(ctx context.Context, uri, name string, pts [][2]int) (serving.Feature, error) {
	var (
		xl   = xlog.FromContextSafe(ctx)
		item serving.Feature
	)

	var face serving.EvalFaceDetection
	if pts == nil {
		fdResp, err := s.FaceDetect.Eval(ctx, serving.EvalFaceDetectReq{Data: struct {
			URI string `json:"uri"`
		}{URI: uri}})
		if err != nil {
			xl.Errorf("call facex-detect uri %s, error: %v", uri, err)
			return item, err
		}
		fdResp.Result.Detections = s.filterSmallFace(fdResp.Result.Detections)
		if len(fdResp.Result.Detections) != 1 {
			xl.Warnf("not one face, uri: %s, face num: %d", uri, len(fdResp.Result.Detections))
			return item, errors.New("not one face detected")
		}
		face = fdResp.Result.Detections[0]
	} else {
		face.Pts = pts
	}
	var ffReq serving.EvalFaceReq
	ffReq.Data.URI = uri
	ffReq.Data.Attribute.Pts = face.Pts
	ffResp, err := s.FaceFeatureV2.Eval(ctx, ffReq)
	if err != nil {
		xl.Errorf("get face feature failed. %s %v", uri, err)
		return item, err
	}
	item.Value = ffResp
	item.Name = name
	return item, nil
}

////////////////////////////////////////////////////////////////////////////////
/*
 * Group Service
 */

// -----------------------------------------------------------------------------------
// initGroups()
//	- try to initialize all the groups when server start up
func (s *Service) initGroups(ctx context.Context) (err error) {
	xl := xlog.FromContextSafe(ctx)
	groups, err := s.AllGroups(ctx)
	if err != nil {
		xl.Errorf("initGroups: manager.AllGroups error: %s", err.Error())
		return
	}

	for _, group := range groups {
		if group.Version >= 0 && group.State != GroupUnknown && group.Name != "" {

			gInfo, e := s.FeatureSearch.Get(ctx, group.Name)
			if e != nil {
				xl.Errorf("get face group %s failed. err: %v", group.Name, e)
				err = e
				return
			}

			if gInfo.Version == group.Version {
				continue
			}

			if gInfo.Version != 0 {
				if err = s.FeatureSearch.Destroy(ctx, group.Name); err != nil {
					xl.Errorf("fail to destroy feature group %s, err: %v", group.Name, err)
					return
				}
			}

			req := serving.FSCreateReq{Name: group.Name, Size: group.Capacity, Precision: group.Precision, Dimension: group.Dimension, Version: group.Version}
			err = s.FeatureSearch.Create(ctx, req)
			if err != nil {
				xl.Errorf("fail to create feature group %s, size: %d, precision: %d, dimension: %d,version: %d, err: %v", group.Name, group.Capacity, group.Precision, group.Dimension, group.Version, err)
				return
			}

			if err = s.IterFeature(ctx, group.Name, s.FeatureSearch.Add); err != nil {
				xl.Errorf("fail to iter add features to group %s, err: %s", group.Name, err.Error())
				return
			}

			if err = s.FeatureSearch.UpdateState(ctx, group.Name, GroupInitialized); err != nil {
				xl.Errorf("fail to update group %s to state GroupInitialized, err: %s", group.Name, err.Error())
			}
		}
	}

	return
}

// -----------------------------------------------------------------------------------
type postFaceGroup_NewReq struct {
	CmdArgs   []string
	Size      int `json:"size"`
	Dimension int `json:"dimension,omitempty"`
	Precision int `json:"precision,omitempty"`
}

func (s *Service) PostFaceGroup_New(ctx context.Context, args *postFaceGroup_NewReq, env *restrpc.Env) (err error) {
	xl, ctx := s.initContext(ctx, env)

	var (
		group = args.CmdArgs[0]
	)

	if len(group) == 0 {
		return httputil.NewError(http.StatusBadRequest, "empty face group name")
	}

	_, err = s.GetGroup(ctx, group)
	if err == nil {
		return ErrGroupExist
	}
	if err != ErrGroupNotExist {
		return
	}

	req := serving.FSCreateReq{Name: group, Size: args.Size, Precision: args.Precision, Dimension: args.Dimension, State: GroupInitialized}
	err = s.FeatureSearch.Create(ctx, req)
	if err != nil {
		xl.Errorf("fail to create feature group %s, size: %d, precision: %d, dimension: %d, err: %v", group, args.Size, args.Precision, args.Dimension, err)
		return
	}
	// new set in db
	if err = s.AddGroup(ctx, group, args.Size); err != nil {
		xl.Errorf("PostFaceGroup_New: %s", err.Error())
		return
	}
	xl.Debugf("Create feature group %s, size: %d, precision: %d, dimension: %d", group, args.Size, args.Precision, args.Dimension)
	return
}

// -----------------------------------------------------------------------------------
func (s *Service) PostFaceGroup_Remove(ctx context.Context, args *BaseReq, env *restrpc.Env) (err error) {
	xl, ctx := s.initContext(ctx, env)
	var (
		group = args.CmdArgs[0]
	)

	if len(group) == 0 {
		return httputil.NewError(http.StatusBadRequest, "empty face group name")
	}

	if err = s.DeleteGroup(ctx, group); err != nil {
		xl.Errorf("PostFaceGroup_Remove: manager DeleteGroup error: %s", err.Error())
		return
	}

	if err = s.FeatureSearch.Destroy(ctx, group); err != nil {
		xl.Errorf("PostFaceGroup_Remove: call feature-search.Destroy failed, error: %v", err)
		return
	}
	return
}

// -----------------------------------------------------------------------------------
type getFaceGroupResp struct {
	Groups []string `json:"groups"`
}

func (s *Service) GetFaceGroup(ctx context.Context, args *BaseReq, env *restrpc.Env) (resp getFaceGroupResp, err error) {
	xl, ctx := s.initContext(ctx, env)
	groups, err := s.AllGroups(ctx)
	if err != nil {
		xl.Errorf("GetFaceGroup: manager.AllGroups error: %s", err.Error())
		return
	}
	for _, group := range groups {
		resp.Groups = append(resp.Groups, group.Name)
	}
	xl.Debugf("GetFaceGroup: get all groups %v", resp.Groups)
	return
}

// -----------------------------------------------------------------------------------
type postFaceGroup_AddReq struct {
	CmdArgs []string
	Data    []struct {
		URI       string `json:"uri"`
		Attribute struct {
			Name string   `json:"name,omitempty"`
			Pts  [][2]int `json:"pts,omitempty"`
			ID   string   `json:"id,omitempty"`
		} `json:"attribute,omitempty"`
	} `json:"data"`
}

type postFaceGroup_AddResp struct {
	Faces []string `json:"faces"`
}

func (s *Service) PostFaceGroup_Add(ctx context.Context, args *postFaceGroup_AddReq, env *restrpc.Env) (resp postFaceGroup_AddResp, err error) {
	xl, ctx := s.initContext(ctx, env)
	var (
		group    = args.CmdArgs[0]
		waiter   = sync.WaitGroup{}
		lock     sync.Mutex
		features = make([]serving.Feature, 0, len(args.Data))
	)

	if len(group) == 0 {
		err = httputil.NewError(http.StatusBadRequest, "empty face group name")
		return
	}

	resp.Faces = make([]string, len(args.Data))

	for i, item := range args.Data {
		waiter.Add(1)
		go func(ctx context.Context, index int, uri, name, id string, pts [][2]int) {
			defer waiter.Done()

			item, e := s.parseFace(ctx, uri, name, pts)
			if e != nil {
				err = e
				return
			}
			if id == "" {
				item.ID = xlog.GenReqId()[2:14]
			} else {
				item.ID = id
			}
			lock.Lock()
			defer lock.Unlock()
			features = append(features, item)
			resp.Faces[index] = item.ID
		}(ctx, i, item.URI, item.Attribute.Name, item.Attribute.ID, item.Attribute.Pts)

	}
	waiter.Wait()
	if err != nil {
		xl.Errorf("PostFaceGroup_Add: parseFace failed, error: %s", err.Error())
		return
	}
	if err = s.FeatureSearch.Add(ctx, serving.FSAddReq{Name: group, Features: features}); err != nil {
		xl.Errorf("PostFaceGroup_Add: call feature-search.Add failed, error: %v", err)
		return
	}

	if err = s.AddFeatures(ctx, group, features); err != nil {
		xl.Errorf("PostFaceGroup_New: manager.AddFeatures error: %s", err.Error())
		return
	}
	xl.Debugf("PostFaceGroup_Add: add faces %v", resp.Faces)
	return
}

// -----------------------------------------------------------------------------------
type postFaceGroup_DeleteReq struct {
	CmdArgs []string
	Faces   []string `json:"faces"`
}

func (s *Service) PostFaceGroup_Delete(ctx context.Context, args *postFaceGroup_DeleteReq, env *restrpc.Env) (err error) {
	xl, ctx := s.initContext(ctx, env)
	var (
		group = args.CmdArgs[0]
	)

	if len(group) == 0 {
		err = httputil.NewError(http.StatusBadRequest, "empty face group name")
		return
	}
	if len(args.Faces) == 0 {
		return
	}

	if err = s.DeleteFeatures(ctx, group, args.Faces); err != nil {
		xl.Errorf("PostFaceGroup_Delete: manager.DeleteFeatures error: %s", err.Error())
		return
	}

	if _, err = s.FeatureSearch.Delete(ctx, serving.FSDeleteReq{Name: group, IDs: args.Faces}); err != nil {
		xl.Errorf("PostFaceGroup_Delete: call feature-search.Add failed, error: %v", err)
		return
	}
	return
}

// -----------------------------------------------------------------------------------
type faceSearchReq struct {
	Data struct {
		URI string `json:"uri"`
	} `json:"data"`
	Params struct {
		SearchLimit int      `json:"search_limit"`
		FaceLimit   int      `json:"face_limit"`
		Groups      []string `json:"groups"`
		Cluster     string   `json:"cluster"`
	} `json:"params"`
}

// FaceSearchResp ...
type FaceSearchResp struct {
	Code    int              `json:"code"`
	Message string           `json:"message"`
	Result  FaceSearchResult `json:"result"`
}

// FaceSearchResult ...
type FaceSearchResult struct {
	Review     bool               `json:"review"`
	Detections []FaceSearchDetail `json:"detections"`
}

type FaceDetectBox struct {
	Pts   [][2]int `json:"pts"`
	Score float32  `json:"score"`
}

type Value struct {
	Name  string  `json:"name,omitempty"`
	ID    string  `json:"id"`
	Score float32 `json:"score"`
}

const (
	GroupSearchTypeUnknown = iota
	GroupSearchTypeGroup
	GroupSearchTypeCluster
)

type GroupSearh struct {
	Values []Value `json:"values"`
	ID     string  `json:"id"`
	Type   int     `json:"type"`
}

// FaceSearchDetail ...
type FaceSearchDetail struct {
	BoundingBox FaceDetectBox `json:"boundingBox"`
	Groups      []GroupSearh  `json:"groups"`
}

func (s *Service) search(ctx context.Context, groups []string, req *faceSearchReq, env *restrpc.Env, logPrefix string) {

	var (
		resp   FaceSearchResp
		fs     []serving.FSSearchReq
		waiter sync.WaitGroup
		lock   sync.Mutex
		timers = make(map[string]time.Duration, 0)
		start  = time.Now()
	)
	resp.Result.Detections = make([]FaceSearchDetail, 0)

	xl, ctx := s.initContext(ctx, env)
	if req.Params.FaceLimit == 0 {
		req.Params.FaceLimit = s.FaceLimit
	}
	timers["preprocess"] = time.Since(start)
	start = time.Now()

	if strings.TrimSpace(req.Data.URI) == "" {
		xl.Warnf("%s: empty data.uri", logPrefix)
		httputil.ReplyErr(env.W, http.StatusBadRequest, "empty data.uri")
		return
	}

	fdResp, err := s.FaceDetect.Eval(ctx, serving.EvalFaceDetectReq{Data: struct {
		URI string `json:"uri"`
	}{URI: req.Data.URI}})
	if err != nil {
		xl.Errorf("%s: call face detect %s failed, error: %v, resp: %#v", logPrefix, req.Data.URI, err, fdResp)
		httputil.ReplyErr(env.W, http.StatusInternalServerError, err.Error())
		return
	}
	timers["facex-detect"] = time.Since(start)
	start = time.Now()

	{
		// 过滤过小的脸
		fdResp.Result.Detections = s.filterSmallFace(fdResp.Result.Detections)
	}

	if len(fdResp.Result.Detections) > req.Params.FaceLimit {
		features := fdResp.Result.Detections
		sort.Slice(features, func(i, j int) bool { return features[i].Score > features[j].Score })
		fdResp.Result.Detections = features[:req.Params.FaceLimit]
	}

	if len(fdResp.Result.Detections) == 0 {
		httputil.Reply(env.W, http.StatusOK, resp)
		return
	}

	resp.Result.Detections = make([]FaceSearchDetail, len(fdResp.Result.Detections))

	if len(groups) > 0 {
		for _, group := range groups {
			fs = append(fs, serving.FSSearchReq{
				Name:     group,
				Features: make([]serving.Feature, len(fdResp.Result.Detections)),
				Limit:    req.Params.SearchLimit})
		}
	} else {
		// only cluster, no group search
		fs = append(fs, serving.FSSearchReq{
			Features: make([]serving.Feature, len(fdResp.Result.Detections)),
			Limit:    req.Params.SearchLimit,
		})
	}
	for i, d := range fdResp.Result.Detections {
		resp.Result.Detections[i].BoundingBox.Score = d.Score
		resp.Result.Detections[i].BoundingBox.Pts = d.Pts
		resp.Result.Detections[i].Groups = make([]GroupSearh, len(groups))
		waiter.Add(1)
		ctx2 := xlog.NewContext(ctx, xlog.FromContextSafe(ctx).Spawn())
		go func(ctx context.Context, index int, detection serving.EvalFaceDetection) {
			defer waiter.Done()
			var ff serving.EvalFaceReq
			ff.Data.URI = req.Data.URI
			ff.Data.Attribute.Pts = detection.Pts
			feature, e := s.FaceFeatureV2.Eval(ctx, ff)
			if e != nil {
				xl.Errorf("%s: call face featurev2 failed, error: %v", logPrefix, e)
				lock.Lock()
				if err == nil {
					err = e
				}
				defer lock.Unlock()
				return
			}
			for _, f := range fs {
				f.Features[index].Value = feature
			}
		}(ctx2, i, d)
	}
	waiter.Wait()
	if err != nil {
		httputil.ReplyErr(env.W, http.StatusInternalServerError, err.Error())
		return
	}
	timers["facex-featureV2"] = time.Since(start)
	start = time.Now()
	var catched bool
	for index, group := range groups {
		waiter.Add(1)
		ctx2 := xlog.NewContext(ctx, xlog.FromContextSafe(ctx).Spawn())
		go func(ctx context.Context, index int, group string) {
			defer waiter.Done()
			fs[index].Threshold = s.SearchThreshold
			fsResp, err := s.FeatureSearch.Search(ctx, fs[index])
			if err != nil {
				xl.Errorf("%s: call feature search failed, error: %v", logPrefix, err)
				httputil.ReplyErr(env.W, http.StatusInternalServerError, err.Error())
				return
			}
			for i, result := range fsResp.SearchResults {
				resp.Result.Detections[i].Groups[index].Values = make([]Value, 0)
				for _, r := range result {
					if r.Score > s.SearchThreshold {
						ff, err := s.GetFeature(ctx, r.ID)
						if err != nil {
							xl.Errorf("PostFaceGroup_Search: manager.GetFeature for id %s failed, err: %s", r.ID, err.Error())
							return
						}
						value := Value{
							ID:    r.ID,
							Score: r.Score,
							Name:  ff.Name,
						}
						catched = true
						resp.Result.Detections[i].Groups[index].Values = append(resp.Result.Detections[i].Groups[index].Values, value)
						resp.Result.Detections[i].Groups[index].Type = GroupSearchTypeGroup
					}
				}
				resp.Result.Detections[i].Groups[index].ID = group
			}
		}(ctx2, index, group)
	}
	waiter.Wait()
	if len(groups) > 0 {
		timers["feature-search"] = time.Since(start)
	}

	if !catched && len(req.Params.Cluster) > 0 {
		start = time.Now()
		fsReq := fs[0]
		fsReq.Name = req.Params.Cluster
		fsReq.Limit = 1
		fsReq.Threshold = s.ClusterThreshold
		fsResp, err := s.FeatureSearch.Search(ctx, fsReq)
		if err != nil {
			xl.Errorf("%s: call cluster (%s) feature search failed, error: %v", fsReq.Name, logPrefix, err)
			httputil.ReplyErr(env.W, http.StatusInternalServerError, err.Error())
			return
		}
		for index, result := range fsResp.SearchResults {
			var gs GroupSearh
			if len(result) == 0 || result[0].Score < s.ClusterThreshold {
				id := xlog.GenReqId()[2:14]
				feature := serving.Feature{ID: id, Value: fsReq.Features[index].Value}
				if err = s.FeatureSearch.Add(ctx, serving.FSAddReq{Name: req.Params.Cluster, Features: []serving.Feature{feature}}); err != nil {
					xl.Errorf("%s: call feature-search.Add failed, error: %v", logPrefix, err)
					return

				}
				gs.ID = req.Params.Cluster
				gs.Type = GroupSearchTypeCluster
				gs.Values = append(gs.Values, Value{ID: id, Score: 1.0})
			} else {
				r := result[0]
				gs.ID = req.Params.Cluster
				gs.Type = GroupSearchTypeCluster
				gs.Values = append(gs.Values, Value{ID: r.ID, Score: r.Score})
			}
			resp.Result.Detections[index].Groups = append(resp.Result.Detections[index].Groups, gs)
			continue
		}
		timers["feature-clusters"] = time.Since(start)
	}

	if catched || (len(req.Params.Cluster) > 0 && resp.Result.Detections[0].Groups[len(resp.Result.Detections[0].Groups)-1].Values[0].Score == 1.0) {
		xl.Debugf("%s: cluster (%s), resp %#v, duration (%v)", logPrefix, req.Params.Cluster, resp, timers)
	}
	httputil.Reply(env.W, http.StatusOK, resp)
}
func (s *Service) PostFaceGroup_Search(
	ctx context.Context, args *BaseReq, env *restrpc.Env,
) {

	xl, ctx := s.initContext(ctx, env)

	var (
		groups = strings.Split(args.CmdArgs[0], ",")
		req    faceSearchReq
	)

	switch ct := env.Req.Header.Get(CONTENT_TYPE); {
	case IsJsonContent(ct):
		bs, err := ioutil.ReadAll(args.ReqBody)
		defer args.ReqBody.Close()
		if err != nil {
			xl.Warnf("read requests body failed. %v", err)
			httputil.ReplyErr(env.W, http.StatusBadRequest, err.Error())
			return
		}
		if err = json.Unmarshal(bs, &req); err != nil {
			xl.Warnf("unmarshal face search request failed, %v", err)
			httputil.ReplyErr(env.W, http.StatusBadRequest, "parse task request failed")
			return
		}

	case ct == CT_STREAM:
		bs, err := ioutil.ReadAll(args.ReqBody)
		defer args.ReqBody.Close()
		if err != nil {
			xl.Warnf("read requests body failed. %v", err)
			httputil.ReplyErr(env.W, http.StatusBadRequest, err.Error())
			return
		}
		req.Data.URI = "data:application/octet-stream;base64," + string(base64.StdEncoding.EncodeToString(bs))

	default:
		xl.Warnf("PostFaceGroup_Search: bad content type: %s", ct)
		httputil.ReplyErr(env.W, http.StatusBadRequest, "wrong content type")
		return
	}

	s.search(ctx, groups, &req, env, "PostFaceGroup_Search")
	return
}

////////////////////////////////////////////////////////////////////////////////
