package faceg

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"strconv"
	"strings"
	"sync"
	"time"

	"gopkg.in/mgo.v2"

	"github.com/pkg/errors"
	"github.com/qiniu/http/httputil.v1"
	"github.com/qiniu/xlog.v1"
	"qbox.us/dht"
	"qiniu.com/auth/authstub.v1"

	ahttp "qiniu.com/argus/argus/com/http"
	URI "qiniu.com/argus/argus/com/uri"
	"qiniu.com/argus/argus/com/util"
	FG "qiniu.com/argus/feature_group"
	STS "qiniu.com/argus/sts/client"
	"qiniu.com/argus/utility"
	"qiniu.com/argus/utility/evals"
)

const (
	MAX_FACE_SEARCH_LIMIT  = 10000
	MAX_FACE_DESC_LENGTH   = 4096
	MAX_SEARCH_GROUP_COUNT = 5
)

var NoChargeUtype = uint32(0)

type Config struct {
	Saver *FG.SaverConfig `json:"saver"`
}

type FaceGroupService struct {
	Config
	_FaceGroupManager

	FG.FeatureAPIs
	FG.Saver

	dht.Interface
	*sync.RWMutex
}

func NewFaceGroupService(conf Config, sts STS.Client, manager _FaceGroupManager, featureAPIs FG.FeatureAPIs) (*FaceGroupService, func(context.Context, dht.NodeInfos)) {
	s := &FaceGroupService{
		Config:            conf,
		_FaceGroupManager: manager,
		FeatureAPIs:       featureAPIs,
		Interface:         dht.NewCarp(dht.NodeInfos{}),
		RWMutex:           new(sync.RWMutex),
		Saver: func() FG.Saver {
			if conf.Saver != nil {
				return FG.NewKodoSaver(*conf.Saver, sts)
			}
			return nil
		}(),
	}
	return s, s.update
}

func (s *FaceGroupService) update(ctx context.Context, nodes dht.NodeInfos) {
	s.Lock()
	defer s.Unlock()
	s.Interface.Setup(nodes)
}

func (s *FaceGroupService) route(key []byte, ttl int) dht.RouterInfos {
	s.RLock()
	defer s.RUnlock()
	return s.Interface.Route(key, ttl)
}

func (s *FaceGroupService) parseFace(
	ctx context.Context, uri, id, name string, uid, utype uint32, api FaceGFeatureAPI, mode string,
) (_FaceItem, []byte, error) {

	var (
		xl   = xlog.FromContextSafe(ctx)
		item = _FaceItem{
			Name: name,
		}
	)

	var req evals.SimpleReq
	req.Data.URI = uri
	t1 := time.Now()
	dResp, err := api.IFaceDetect.Eval(ctx, req, uid, utype)
	if err != nil {
		_ClientTimeHistogram("face-detect", httputil.DetectCode(err)).Observe(float64(time.Since(t1) / time.Second))
		xl.Errorf("call facex-detect error: %s %v", uri, err)
		return item, nil, err
	}
	_ClientTimeHistogram("face-detect", 200).Observe(float64(time.Since(t1) / time.Second))

	xl.Infof("face detect: %#v", dResp.Result)

	one, err := checkFaceDetectionResp(xl, dResp, mode, uri)
	if err != nil {
		return item, nil, err
	}
	var fReq evals.FaceReq
	fReq.Data.URI = uri
	fReq.Data.Attribute.Pts = one.Pts
	ff, err := api.IFaceFeature.Eval(ctx, fReq, uid, utype)
	if err != nil {
		xl.Errorf("get face feature failed. %s %v", uri, err)
		return item, nil, err
	}

	xl.Infof("face feature: %d", len(ff))

	if len(id) > 0 {
		item.ID = id
	} else {
		item.ID = xlog.GenReqId()
	}

	item.BoundingBox = utility.FaceDetectBox{
		Pts:   one.Pts,
		Score: one.Score,
	}

	return item, ff, nil
}

func (s *FaceGroupService) addFace(
	ctx context.Context, data []FaceGroupAddData, fg _FaceGroup, api FaceGFeatureAPI, uid, utype uint32, gid string,
) (ret *FaceGroupAddResp) {
	var (
		xl     = xlog.FromContextSafe(ctx)
		waiter = sync.WaitGroup{}
		lock   = new(sync.Mutex)

		faces    = make([]_FaceItem, 0, len(data))
		features = make([][]byte, 0, len(data))
		idx      = make([]int, 0, len(data))
	)

	ret = &FaceGroupAddResp{}
	ret.Faces = make([]string, len(data))
	ret.Attributes = make([]*struct {
		BoundingBox utility.FaceDetectBox `json:"bounding_box"`
	}, len(data))
	ret.Errors = make([]*struct {
		Code    int    `json:"code"`
		Message string `json:"message"`
	}, len(data))
	for i, item := range data {
		waiter.Add(1)
		go func(ctx context.Context, index int, uri, id, name string, mode string, desc json.RawMessage) {
			defer waiter.Done()

			if len(desc) > MAX_FACE_DESC_LENGTH {
				lock.Lock()
				defer lock.Unlock()
				ret.Errors[index] = new(struct {
					Code    int    `json:"code"`
					Message string `json:"message"`
				})
				ret.Errors[index].Code = http.StatusNotAcceptable
				ret.Errors[index].Message = "desc too long"
				return
			}

			xl := xlog.FromContextSafe(ctx)
			item, ff, err := s.parseFace(ctx, uri, id, name, uid, utype, api, mode)
			if err != nil {
				lock.Lock()
				defer lock.Unlock()
				ret.Errors[index] = new(struct {
					Code    int    `json:"code"`
					Message string `json:"message"`
				})
				ret.Errors[index].Code = httputil.DetectCode(err)
				ret.Errors[index].Message = err.Error()
				return
			}
			item.Desc = desc

			var url string
			if s.Saver != nil {
				f := func(ctx context.Context) error {
					var _err error
					url, _err = s.Saver.Save(ctx, uid, gid, uri)
					return _err
				}
				err := callRetry(ctx, f)
				if err != nil {
					xl.Infof("SAVE Error: %s %v", url, err)
				} else {
					xl.Infof("SAVE: %s", url)
				}
			}
			item.Backup = url
			if len(item.Backup) == 0 {
				xl.Infof("empty backup url. %s %s %s", uri, id, name)
			}

			lock.Lock()
			defer lock.Unlock()
			faces = append(faces, item)
			features = append(features, ff)
			idx = append(idx, index)
			ret.Faces[index] = item.ID
			ret.Attributes[index] = new(struct {
				BoundingBox utility.FaceDetectBox `json:"bounding_box"`
			})
			ret.Attributes[index].BoundingBox = item.BoundingBox

		}(util.SpawnContext(ctx), i,
			item.URI, item.Attribute.ID, item.Attribute.Name, item.Attribute.Mode, item.Attribute.Desc)
	}

	waiter.Wait()
	errs := fg.Add(ctx, faces, features)
	for i := range errs {
		if err := errs[i]; err != nil {
			xl.Errorf("ad face failed. %#v %v", faces[i], err)

			index := idx[i]
			ret.Errors[index] = new(struct {
				Code    int    `json:"code"`
				Message string `json:"message"`
			})
			ret.Errors[index].Code = httputil.DetectCode(err)
			ret.Errors[index].Message = err.Error()
			ret.Faces[index] = ""
			ret.Attributes[index] = nil
		}
	}
	return
}

func (s *FaceGroupService) GetFaceGroup(
	ctx context.Context, env *authstub.Env,
) (ret struct {
	Code    int      `json:"code"`
	Message string   `json:"message"`
	Result  []string `json:"result"`
}, err error) {

	var (
		uid = env.UserInfo.Uid
	)
	ctx, xl := util.CtxAndLog(ctx, env.W, env.Req)

	_RequestGauge("all").Inc()
	defer func(t1 time.Time) {
		var code = 200
		if err != nil {
			code = httputil.DetectCode(err)
		}
		_ResponseTimeHistogram("all", code).Observe(float64(time.Since(t1) / time.Second))
	}(time.Now())

	ids, err := s._FaceGroupManager.All(ctx, uid)
	if err != nil {
		xl.Errorf("all group failed. %v", err)
		return
	}
	xl.Infof("all group done. %d", len(ids))
	ret.Result = ids
	return
}

func (s *FaceGroupService) GetFaceGroup_(
	ctx context.Context,
	req *struct {
		CmdArgs []string
	},
	env *authstub.Env,
) (ret struct {
	Code    int    `json:"code"`
	Message string `json:"message"`
	Result  []struct {
		ID    string `json:"id"`
		Value struct {
			Name string `json:"name"`
		} `json:"value"`
	} `json:"result"`
}, err error) {

	var (
		uid = env.UserInfo.Uid
		id  = req.CmdArgs[0]
	)
	ctx, xl := util.CtxAndLog(ctx, env.W, env.Req)

	_RequestGauge("list").Inc()
	defer func(t1 time.Time) {
		var code = 200
		if err != nil {
			code = httputil.DetectCode(err)
		}
		_ResponseTimeHistogram("list", code).Observe(float64(time.Since(t1) / time.Second))
	}(time.Now())

	fg, err := s._FaceGroupManager.Get(ctx, uid, id)
	if err != nil {
		xl.Errorf("get group failed. %s %v", id, err)
		return
	}
	itemsIter, err := fg.Iter(ctx)
	if err != nil {
		xl.Errorf("get face iter failed. %s %v", id, err)
		return
	}
	xl.Infof("get face iter done")

	defer itemsIter.Close()
	for {
		item, ok := itemsIter.Next(ctx)
		if !ok {
			break
		}

		ret.Result = append(ret.Result,
			struct {
				ID    string `json:"id"`
				Value struct {
					Name string `json:"name"`
				} `json:"value"`
			}{
				ID: item.ID,
				Value: struct {
					Name string `json:"name"`
				}{
					Name: item.Name,
				},
			},
		)
	}
	err = itemsIter.Error()
	if err != nil {
		xl.Errorf("face iter failed. %s %v", id, err)
		return
	}
	return
}

func (s *FaceGroupService) PostFaceGroup_Remove(
	ctx context.Context,
	req *struct {
		CmdArgs []string
	},
	env *authstub.Env,
) error {
	var (
		uid = env.UserInfo.Uid
		id  = req.CmdArgs[0]
	)

	var err error
	_RequestGauge("remove").Inc()
	defer func(t1 time.Time) {
		var code = 200
		if err != nil {
			code = httputil.DetectCode(err)
		}
		_ResponseTimeHistogram("remove", code).Observe(float64(time.Since(t1) / time.Second))
	}(time.Now())

	ctx, xl := util.CtxAndLog(ctx, env.W, env.Req)
	if err = s._FaceGroupManager.Remove(ctx, uid, id); err != nil {
		xl.Errorf("remove group failed. %s %v", id, err)
	}
	xl.Infof("remove group %s", id)
	return nil
}

func (s *FaceGroupService) PostFaceGroup_Delete(
	ctx context.Context,
	req *struct {
		CmdArgs []string
		Faces   []string `json:"faces"`
	},
	env *authstub.Env,
) error {
	var (
		uid = env.UserInfo.Uid
		id  = req.CmdArgs[0]
	)

	var err error
	_RequestGauge("delete").Inc()
	defer func(t1 time.Time) {
		var code = 200
		if err != nil {
			code = httputil.DetectCode(err)
		}
		_ResponseTimeHistogram("delete", code).Observe(float64(time.Since(t1) / time.Second))
	}(time.Now())

	ctx, xl := util.CtxAndLog(ctx, env.W, env.Req)
	fg, err := s._FaceGroupManager.Get(ctx, uid, id)
	if err != nil {
		xl.Errorf("get group failed. %s %v", id, err)
		return err
	}
	if err = fg.Del(ctx, req.Faces); err != nil {
		xl.Errorf("del faces failed. %d %v", len(req.Faces), err)
		return err
	}
	xl.Infof("group %s delete face", id)
	return nil
}

func (s *FaceGroupService) PostFaceGroup_New(
	ctx context.Context,
	req *FaceGroupAddReq,
	env *authstub.Env,
) (ret *FaceGroupAddResp, err error) {

	var (
		uid   = env.UserInfo.Uid
		utype = env.UserInfo.Utype
		gid   = req.CmdArgs[0]
	)
	ctx, xl := util.CtxAndLog(ctx, env.W, env.Req)

	_RequestGauge("new").Inc()
	defer func(t1 time.Time) {
		var code = 200
		if err != nil {
			code = httputil.DetectCode(err)
		}
		_ResponseTimeHistogram("new", code).Observe(float64(time.Since(t1) / time.Second))
	}(time.Now())

	fv, api, err := s.FeatureAPIs.Current()
	if err != nil {
		return
	}
	fg, err := s._FaceGroupManager.New(ctx, uid, gid, fv)
	if mgo.IsDup(err) {
		err = httputil.NewError(http.StatusBadRequest, `group already exists`)
		return
	}
	if err != nil {
		return
	}
	xl.Infof("create group %s", gid)

	if err = checkFaceMode(req); err != nil {
		return
	}

	ret = s.addFace(ctx, req.Data, fg, api.(FaceGFeatureAPI), uid, utype, gid)
	xl.Infof("group %s add face", gid)
	return
}

func (s *FaceGroupService) PostFaceGroup_Add(
	ctx context.Context,
	req *FaceGroupAddReq,
	env *authstub.Env,
) (ret *FaceGroupAddResp, err error) {

	var (
		uid   = env.UserInfo.Uid
		utype = env.UserInfo.Utype
		gid   = req.CmdArgs[0]
	)

	_RequestGauge("add").Inc()
	defer func(t1 time.Time) {
		var code = 200
		if err != nil {
			code = httputil.DetectCode(err)
		}
		_ResponseTimeHistogram("add", code).Observe(float64(time.Since(t1) / time.Second))
	}(time.Now())

	ctx, xl := util.CtxAndLog(ctx, env.W, env.Req)

	fg, err := s._FaceGroupManager.Get(ctx, uid, gid)
	if err != nil {
		xl.Errorf("get group failed. %s %v", gid, err)
		return
	}

	hid, hub := fg.Hub(ctx)
	_, api, err := s.getFeatureAPI(ctx, hub, hid)
	if err != nil {
		xl.Errorf("get feature api failed. %s %v", gid, err)
		return
	}

	if err = checkFaceMode(req); err != nil {
		return
	}

	ret = s.addFace(ctx, req.Data, fg, api.(FaceGFeatureAPI), uid, utype, gid)
	xl.Infof("group %s add face", gid)
	return
}

//----------------------------------------------------------------------------//

func (s *FaceGroupService) PostFaceGroup_Feature(
	ctx context.Context,
	req *struct {
		CmdArgs []string
		ReqBody *FG.SearchKey
	},
	env *authstub.Env,
) {
	ctx, xl := util.CtxAndLog(ctx, env.W, env.Req)

	var err error
	_RequestGauge("feature").Inc()
	defer func(t1 time.Time) {
		var code = 200
		if err != nil {
			code = httputil.DetectCode(err)
		}
		_ResponseTimeHistogram("feature", code).Observe(float64(time.Since(t1) / time.Second))
	}(time.Now())

	xl.Infof("REQ: %#v", req.ReqBody)

	hub := s._FaceGroupManager.Hub(ctx)
	iter, err := hub.Fetch(ctx,
		req.ReqBody.Hid, req.ReqBody.Version,
		req.ReqBody.From, req.ReqBody.To,
	)

	if err != nil {
		code, msg := httputil.DetectError(err)
		httputil.ReplyErr(env.W, code, msg)
		return
	}

	_, api, err := s.getFeatureAPI(ctx, hub, req.ReqBody.Hid)
	if err != nil {
		code, msg := httputil.DetectError(err)
		httputil.ReplyErr(env.W, code, msg)
		return
	}

	facegAPI := api.(FaceGFeatureAPI)
	h := env.W.Header()
	h.Set("Content-Length", strconv.FormatInt(int64(req.ReqBody.To-req.ReqBody.From)*int64(facegAPI.FeatureLength)*4, 10))
	h.Set("Content-Type", "application/octet-stream")
	env.W.WriteHeader(200)

	xl.Infof("%#v", req.ReqBody)

	defer iter.Close()
	for {
		bs, ok := iter.Next(ctx)
		if !ok {
			break
		}
		// xl.Debugf("%#v %d", req.ReqBody, len(bs))

		// search worker always expects Little-Endian
		if facegAPI.ReserveByteOrder == true && facegAPI.FeatureByteOrder == FG.BigEndian {
			bs = FG.BigEndianToLittleEndian(bs)
		}
		env.W.Write(bs)
	}
}

func (s *FaceGroupService) postFaceGroupSearch(
	ctx context.Context, uid uint32, gids []string, ffs [][]byte, lengths []int, thresholds []float32, limit, blockSize int,
) (ret []FaceGroupsSearchDetailValue, err error) {
	blocks := make([]_FGSearchBlock, 0)
	for i, gid := range gids {
		fg, err := s._FaceGroupManager.Get(ctx, uid, gid)
		if err != nil {
			return nil, errors.Wrapf(err, "_FaceGroupManager.Get %v %v", uid, gid)
		}

		hid, hub := fg.Hub(ctx)
		fbs, err := hub.All(ctx, hid, blockSize)
		if err != nil {
			return nil, errors.Wrapf(err, "hub.All %v %v", hid, blockSize)
		}

		featureStr := base64.StdEncoding.EncodeToString(ffs[i])
		for _, fb := range fbs {
			blocks = append(blocks, _FGSearchBlock{
				FeatureBlock:  fb,
				FaceFeature:   &featureStr,
				FeatureLength: lengths[i],
				Threshold:     thresholds[i],
				_FGSearchGroupHub: _FGSearchGroupHub{
					FaceGroup: fg,
					Gid:       gid,
					Hub:       hub,
					Hid:       hid,
				},
			})
		}
	}

	var (
		xl      = xlog.FromContextSafe(ctx)
		results = make([]_FGSearchResultItem, 0)
		lock    sync.Mutex
		wg      sync.WaitGroup
	)
	for _, block := range blocks {
		wg.Add(1)
		go func(ctx context.Context, sb _FGSearchBlock) {
			defer wg.Done()

			var sReq = FG.SearchReq{
				Key: FG.SearchKey{
					Hid:     sb.FeatureBlock.Hid,
					Version: sb.FeatureBlock.Ver,
					From:    sb.FeatureBlock.From,
					To:      sb.FeatureBlock.To,
				},
				Threshold: sb.Threshold,
				Limit:     limit,
				Features:  *sb.FaceFeature,
				Length:    sb.FeatureLength,
			}

			workers := s.route([]byte(sReq.Key.Key()), 1)
			if len(workers) == 0 || len(workers[0].Host) == 0 {
				xl.Errorf("no search worker found. %#v", sReq.Key)
				return
			}
			searcher := FG.SearchAPI{
				Host:    workers[0].Host,
				Path:    "/v1/search",
				Timeout: time.Second * 60,
			}
			sResp, err := searcher.Search(ctx, sReq)
			if err != nil {
				xl.Errorf("get face match failed. %v", err)
				return
			}
			if len(sResp) == 0 {
				xl.Errorf("get face match failed. %v", sResp)
				return
			}

			lock.Lock()
			defer lock.Unlock()

			temp := results
			results = make([]_FGSearchResultItem, 0, limit)
			i, j := 0, 0
			for len(results) < limit && i < len(temp) && j < len(sResp[0].Items) {
				if temp[i].Score > sResp[0].Items[j].Score {
					results = append(results, temp[i])
					i++
				} else if temp[i].Score < sResp[0].Items[j].Score {
					results = append(results, _FGSearchResultItem{
						_FGSearchGroupHub: sb._FGSearchGroupHub,
						SearchResultItem:  sResp[0].Items[j],
					})
					j++
				} else {
					results = append(results, temp[i])
					results = append(results, _FGSearchResultItem{
						_FGSearchGroupHub: sb._FGSearchGroupHub,
						SearchResultItem:  sResp[0].Items[j],
					})
					i++
					j++
				}
			}
			for ; len(results) < limit && i < len(temp); i++ {
				results = append(results, temp[i])
			}
			for ; len(results) < limit && j < len(sResp[0].Items); j++ {
				results = append(results, _FGSearchResultItem{
					_FGSearchGroupHub: sb._FGSearchGroupHub,
					SearchResultItem:  sResp[0].Items[j],
				})
			}

		}(util.SpawnContext(ctx), block)
	}
	wg.Wait()

	for _, _ret := range results {
		fid, err := _ret.Hub.Find(ctx, _ret.Hid, _ret.Version, _ret.Index)
		if err != nil {
			xl.Warnf("fid not exist. %v %v %v", _ret.Hid, _ret.Version, _ret.Index)
			continue
		}
		item, err := _ret.FaceGroup.Get(ctx, string(fid))
		if err == os.ErrNotExist {
			xl.Warnf("not exist. %s %d %d %s", _ret.Hid, _ret.Version, _ret.Index, fid)
			continue
		}

		score := _ret.Score
		if score < 0 {
			score = 0
		}
		ret = append(ret, FaceGroupsSearchDetailValue{
			Name:        item.Name,
			ID:          item.ID,
			BoundingBox: item.BoundingBox,
			Desc:        item.Desc,
			Score:       score,
			Group:       _ret.Gid,
		})
	}

	return ret, nil
}

//FaceGroupSearch
//-----------------------------------------------------------//

func (s *FaceGroupService) PostFaceGroup_Search(
	ctx context.Context,
	req *struct {
		CmdArgs []string
		Data    struct {
			URI string `json:"uri"`
		} `json:"data"`
	},
	env *authstub.Env,
) (ret *FaceGroupSearchResp, err error) { //deprecated
	ctx, xl := util.CtxAndLog(ctx, env.W, env.Req)

	v2Req := &FaceGroupsSearchReq{
		Data: req.Data,
		Params: FaceGroupsSearchReqParams{
			Groups: []string{req.CmdArgs[0]},
		},
	}

	v2Ret, err := s.PostFaceGroupsSearch(ctx, v2Req, env)
	if err != nil {
		xl.Error("call PostFaceGroupsSearch failed. %v", err)
		return
	}

	ret = convertSearchRespFromMultiToSingle(v2Ret)
	return
}

func (s *FaceGroupService) PostFaceGroupsSearch(
	ctx context.Context,
	req *FaceGroupsSearchReq,
	env *authstub.Env,
) (ret *FaceGroupsSearchResp, err error) {

	var (
		uid   = env.UserInfo.Uid
		utype = env.UserInfo.Utype
	)
	ctx, xl := util.CtxAndLog(ctx, env.W, env.Req)

	_RequestGauge("search").Inc()
	defer func(t1 time.Time) {
		var code = 200
		if err != nil {
			code = httputil.DetectCode(err)
		}
		_ResponseTimeHistogram("search", code).Observe(float64(time.Since(t1) / time.Second))
		xl.Infof("request total time ====> %v", time.Since(t1))
	}(time.Now())

	if strings.TrimSpace(req.Data.URI) == "" {
		xl.Error("empty data.uri")
		err = httputil.NewError(http.StatusNotAcceptable, "no enough arguments provided")
		return
	}

	if req.Params.Limit > MAX_FACE_SEARCH_LIMIT {
		xl.Errorf("invalid params.limit, max allowed: %d\n", MAX_FACE_SEARCH_LIMIT)
		err = httputil.NewError(http.StatusNotAcceptable, "invalid limit, maximal allowed limit is 10000")
		return
	}

	if req.Params.Threshold < 0 || req.Params.Threshold > 1 {
		xl.Error("invalid params.threshold")
		err = httputil.NewError(http.StatusNotAcceptable, "invalid threshold")
		return
	}

	//remove duplicate group
	groupIds := FG.UniqueStringSlice(req.Params.Groups)
	if len(groupIds) == 0 {
		xl.Error("empty groups")
		err = httputil.NewError(http.StatusNotAcceptable, "groups is empty")
		return
	}
	if len(groupIds) > MAX_SEARCH_GROUP_COUNT {
		xl.Error("too many groups")
		err = httputil.NewError(http.StatusNotAcceptable, fmt.Sprintf("cannot search in more than %d groups at one time", MAX_SEARCH_GROUP_COUNT))
		return
	}

	//get groups
	fgs := make([]_FaceGroup, 0)
	for _, id := range groupIds {
		fg, err := s._FaceGroupManager.Get(ctx, uid, id)
		if err != nil {
			xl.Errorf("get group failed. %s %v", id, err)
			return nil, err
		}
		fgs = append(fgs, fg)
	}

	//get limit
	limit := req.Params.Limit
	// negative limit means want to get all possible satisfied face
	// so set limit to the total face number of the group
	if limit < 0 {
		limit = 0
		for _, v := range fgs {
			n, err := v.Count(ctx)
			if err != nil {
				xl.Errorf("get face count failed. %v", err)
				return nil, err
			}
			limit += n
			//alway set the limit even return want to all the result
			if limit > MAX_FACE_SEARCH_LIMIT {
				limit = MAX_FACE_SEARCH_LIMIT
				break
			}
		}
	}
	//if limit is still 0, set it to 1
	if limit == 0 {
		limit = 1
	}

	//get api & threshold & featureLength of all groups
	facegAPIs := make([]FaceGFeatureAPI, 0)
	thresholds := make([]float32, 0)
	featureLengths := make([]int, 0)
	for i, v := range fgs {
		hid, hub := v.Hub(ctx)
		_, api, err := s.getFeatureAPI(ctx, hub, hid)
		if err != nil {
			xl.Errorf("get feature api failed. %s %v", groupIds[i], err)
			return nil, err
		}
		facegAPI := api.(FaceGFeatureAPI)
		featureLengths = append(featureLengths, facegAPI.FeatureLength)
		facegAPIs = append(facegAPIs, facegAPI)

		threshold := req.Params.Threshold
		if threshold < facegAPI.Threshold {
			threshold = facegAPI.Threshold
		}
		thresholds = append(thresholds, threshold)
	}

	//do face-detect
	var iReq evals.SimpleReq
	iReq.Data.URI = req.Data.URI
	t1 := time.Now()
	//FaceDetect目前没有版本区别=>直接使用第一个group的IFaceDetect去检测人脸
	dResp, err := facegAPIs[0].IFaceDetect.Eval(ctx, iReq, uid, utype)
	xl.Infof("face detect time ====> %v", time.Since(t1))
	if err != nil {
		_ClientTimeHistogram("face-detect", httputil.DetectCode(err)).Observe(float64(time.Since(t1) / time.Second))
		xl.Errorf("call facex-detect error: %s %v", URI.STRING(req.Data.URI), err)
		return nil, err
	}
	_ClientTimeHistogram("face-detect", 200).Observe(float64(time.Since(t1) / time.Second))
	if dResp.Code != 0 && dResp.Code/100 != 2 {
		xl.Errorf("call facex-detect failed: %s %d %s",
			URI.STRING(req.Data.URI), dResp.Code, dResp.Message)
		return nil, errors.New("call facex-detect failed")
	}

	var (
		waiter = sync.WaitGroup{}
		lock   = new(sync.Mutex)
	)
	ret = &FaceGroupsSearchResp{
		Result: FaceGroupsSearchResult{
			Faces: make([]FaceGroupsSearchDetail, 0, len(dResp.Result.Detections)),
		},
	}
	ctxs := make([]context.Context, 0, len(dResp.Result.Detections))
	for _, d := range dResp.Result.Detections {
		waiter.Add(1)
		ctx2 := util.SpawnContext(ctx)
		go func(ctx context.Context, face evals.FaceDetection) {
			defer waiter.Done()
			xl := xlog.FromContextSafe(ctx)

			t1 := time.Now()
			ffs, err1 := s.getFaceFeatures(ctx, facegAPIs, req.Data.URI, face.Pts, uid, utype)
			xl.Infof("face feature time ====> %v", time.Since(t1))
			if err1 != nil {
				lock.Lock()
				err = err1
				lock.Unlock()
				return
			}

			t2 := time.Now()
			_ret, err1 := s.postFaceGroupSearch(ctx, uid, groupIds,
				ffs, featureLengths, thresholds, limit, 64*1024/4, // TODO
			)
			xl.Infof("search time ====> %v", time.Since(t2))
			if err1 != nil {
				_ClientTimeHistogram("search", httputil.DetectCode(err1)).Observe(float64(time.Since(t2) / time.Second))
				xl.Errorf("get face match failed. %v", err1)
				lock.Lock()
				if err == nil {
					err = err1
				}
				lock.Unlock()
				return
			}
			_ClientTimeHistogram("search", 200).Observe(float64(time.Since(t2) / time.Second))

			detail := FaceGroupsSearchDetail{
				BoundingBox: utility.FaceDetectBox{
					Pts:   face.Pts,
					Score: face.Score,
				},
				Faces: _ret,
			}

			lock.Lock()
			defer lock.Unlock()
			ret.Result.Faces = append(ret.Result.Faces, detail)
		}(ctx2, d)
		ctxs = append(ctxs, ctx2)
	}
	waiter.Wait()
	for _, ctx2 := range ctxs {
		xl.Xput(xlog.FromContextSafe(ctx2).Xget())
	}

	if err != nil {
		xl.Errorf("run face group search failed. %v", err)
		return
	}

	xl.Infof("group %s search face", groupIds)

	if len(ret.Result.Faces) == 0 {
		ret.Message = "No valid face info detected"
	}

	if utype != NoChargeUtype {
		util.SetStateHeader(env.W.Header(), "FACE_GROUP_SEARCH", 1)
	}
	return
}

func (s *FaceGroupService) getFeatureAPI(ctx context.Context, hub FG.Hub, hid FG.HubID) (FG.FeatureVersion, FG.FeatureAPI, error) {
	fv, err := hub.FeatureVersion(ctx, hid)
	if err != nil {
		return FG.EmptyFeatureVersion, nil, err
	}

	if len(fv) == 0 { // legacy group
		return s.FeatureAPIs.Default()
	}

	api, err := s.FeatureAPIs.Get(fv)
	return fv, api, err
}

func (s *FaceGroupService) getFaceFeatures(ctx context.Context, apis []FaceGFeatureAPI, uri string, pts [][2]int, uid, utype uint32) (ffs [][]byte, err error) {
	var (
		xl       = xlog.FromContextSafe(ctx)
		waiter   = sync.WaitGroup{}
		lock     = new(sync.Mutex)
		ffMap    = make(map[string][]byte)
		versions []string
		fReq     evals.FaceReq
	)
	fReq.Data.URI = uri
	fReq.Data.Attribute.Pts = pts

	//不同group对应的feature版本可能不同，要分别计算
	//相同的feature版本不重复计算
	for _, api := range apis {
		version := api.FeatureVersion
		//是否已计算过该版本feature
		if FG.StringArrayContains(versions, version) {
			continue
		}

		versions = append(versions, version)
		waiter.Add(1)
		go func(faceApi FaceGFeatureAPI, ver string) {
			defer waiter.Done()
			ff, err1 := faceApi.IFaceFeature.Eval(ctx, fReq, uid, utype)
			if err1 != nil {
				xl.Errorf("get face feature failed. %v", err1)
				lock.Lock()
				if err == nil {
					err = err1
				}
				lock.Unlock()
				return
			}

			// search worker always expects Little-Endian
			if faceApi.ReserveByteOrder && faceApi.FeatureByteOrder == FG.BigEndian {
				ff = FG.BigEndianToLittleEndian(ff)
			}

			lock.Lock()
			ffMap[ver] = ff
			lock.Unlock()
		}(api, version)
	}
	waiter.Wait()

	if err == nil {
		for _, api := range apis {
			ffs = append(ffs, ffMap[api.FeatureVersion])
		}
	}
	return
}

func callRetry(ctx context.Context, f func(context.Context) error) error {
	return ahttp.CallWithRetry(ctx, []int{530}, []func(context.Context) error{f, f})
}

func convertSearchRespFromMultiToSingle(multi *FaceGroupsSearchResp) *FaceGroupSearchResp {
	single := &FaceGroupSearchResp{
		Code:    multi.Code,
		Message: multi.Message,
		Result: FaceGroupSearchResult{
			Review:     multi.Result.Review,
			Detections: make([]FaceGroupSearchDetail, 0),
		},
	}
	for _, v := range multi.Result.Faces {
		detection := FaceGroupSearchDetail{
			BoundingBox: v.BoundingBox,
		}
		if len(v.Faces) > 0 {
			detection.Value = FaceGroupSearchDetailValue{
				Name:        v.Faces[0].Name,
				ID:          v.Faces[0].ID,
				Score:       v.Faces[0].Score,
				Review:      v.Faces[0].Review,
				BoundingBox: v.Faces[0].BoundingBox,
				Desc:        v.Faces[0].Desc,
			}
		}
		single.Result.Detections = append(single.Result.Detections, detection)
	}
	return single
}
