package imageg

import (
	"context"
	"crypto/sha1"
	"encoding/base64"
	"encoding/hex"
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
	"qiniu.com/argus/utility/evals"
)

const (
	MAX_IMAGE_SEARCH_LIMIT = 10000
	MAX_IMAGE_DESC_LENGTH  = 4096
	MAX_SEARCH_GROUP_COUNT = 5
)

var NoChargeUtype = uint32(0)

type Config struct {
	Saver *FG.SaverConfig `json:"saver"`
}

type ImageGroupService struct {
	Config
	_ImageGroupManager

	FG.FeatureAPIs
	FG.Saver

	dht.Interface
	*sync.RWMutex
}

func NewImageGroupServic(conf Config, sts STS.Client, manager _ImageGroupManager, featureAPIs FG.FeatureAPIs) (*ImageGroupService, func(context.Context, dht.NodeInfos)) {
	s := &ImageGroupService{
		Config:             conf,
		_ImageGroupManager: manager,
		FeatureAPIs:        featureAPIs,
		Interface:          dht.NewCarp(dht.NodeInfos{}),
		RWMutex:            new(sync.RWMutex),
		Saver: func() FG.Saver {
			if conf.Saver != nil {
				return FG.NewKodoSaver(*conf.Saver, sts)
			}
			return nil
		}(),
	}
	return s, s.update
}

func (s *ImageGroupService) update(ctx context.Context, nodes dht.NodeInfos) {
	s.Lock()
	defer s.Unlock()
	s.Interface.Setup(nodes)
}

func (s *ImageGroupService) route(key []byte, ttl int) dht.RouterInfos {
	s.RLock()
	defer s.RUnlock()
	return s.Interface.Route(key, ttl)
}

func (s *ImageGroupService) parseImage(
	ctx context.Context, uri, id, label string, uid, utype uint32, api ImageGFeatureAPI,
) (_ImageItem, []byte, error) {

	var (
		xl   = xlog.FromContextSafe(ctx)
		item = _ImageItem{
			Label: label,
		}
	)
	if strings.HasPrefix(uri, URI.DataURIPrefix) {
		var err error
		item.Etag, err = dataURIEtag(uri)
		if err != nil {
			return _ImageItem{}, nil, errors.Wrapf(err, "dataURIEtag %v", uri)
		}
	} else {
		item.URI = uri
	}

	var req evals.SimpleReq
	req.Data.URI = uri
	ff, err := api.IImageFeature.Eval(ctx, req, uid, utype)
	if err != nil {
		xl.Errorf("get image feature failed. %s %v", uri, err)
		return item, ff, err
	}
	xl.Infof("image feature: %d", len(ff))

	if id == "" {
		item.ID = xlog.GenReqId()
	} else {
		item.ID = id
	}

	return item, ff, nil
}

func (s *ImageGroupService) addImage(
	ctx context.Context, data []ImageGroupAddData, ig _ImageGroup, api ImageGFeatureAPI, uid, utype uint32, gid string,
) (ret *ImageGroupAddResp) {
	var (
		xl     = xlog.FromContextSafe(ctx)
		waiter = sync.WaitGroup{}
		lock   = new(sync.Mutex)

		images   = make([]_ImageItem, 0, len(data))
		features = make([][]byte, 0, len(data))
		idx      = make([]int, 0, len(data))
	)

	ret = &ImageGroupAddResp{}
	ret.Images = make([]string, len(data))
	ret.Errors = make([]*struct {
		Code    int    `json:"code"`
		Message string `json:"message"`
	}, len(data))
	for i, item := range data {
		waiter.Add(1)
		go func(ctx context.Context, index int, uri, id, label string, desc json.RawMessage) {
			defer waiter.Done()

			if len(desc) > MAX_IMAGE_DESC_LENGTH {
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
			item, ff, err := s.parseImage(ctx, uri, id, label, uid, utype, api)
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
				xl.Infof("empty backup url. %s %s %s", uri, id, label)
			}

			lock.Lock()
			defer lock.Unlock()
			images = append(images, item)
			features = append(features, ff)
			idx = append(idx, index)
			ret.Images[index] = item.ID

		}(util.SpawnContext(ctx), i, item.URI, item.Attribute.ID, item.Attribute.Label, item.Attribute.Desc)
	}

	waiter.Wait()
	errs := ig.Add(ctx, images, features)
	for i := range errs {
		if err := errs[i]; err != nil {
			xl.Errorf("ad image failed. %#v %v", images[i], err)

			index := idx[i]
			ret.Errors[index] = new(struct {
				Code    int    `json:"code"`
				Message string `json:"message"`
			})
			ret.Errors[index].Code = httputil.DetectCode(err)
			ret.Errors[index].Message = err.Error()
			ret.Images[index] = ""
		}
	}
	return
}

func (s *ImageGroupService) GetImageGroup(
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

	ids, err := s._ImageGroupManager.All(ctx, uid)
	if err != nil {
		xl.Errorf("all group failed. %v", err)
		return
	}
	xl.Infof("all group done. %d", len(ids))
	ret.Result = ids
	return
}

func (s *ImageGroupService) GetImageGroup_(
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
			Label string `json:"label"`
			Etag  string `json:"etag"`
			URI   string `json:"uri"`
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

	ig, err := s._ImageGroupManager.Get(ctx, uid, id)
	if err != nil {
		xl.Errorf("get group failed. %s %v", id, err)
		return
	}
	itemsIter, err := ig.Iter(ctx)
	if err != nil {
		xl.Errorf("get image iter failed. %s %v", id, err)
		return
	}
	xl.Infof("get image iter done")

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
					Label string `json:"label"`
					Etag  string `json:"etag"`
					URI   string `json:"uri"`
				} `json:"value"`
			}{
				ID: item.ID,
				Value: struct {
					Label string `json:"label"`
					Etag  string `json:"etag"`
					URI   string `json:"uri"`
				}{
					Label: item.Label,
					Etag:  item.Etag,
					URI:   item.URI,
				},
			},
		)
	}
	err = itemsIter.Error()
	if err != nil {
		xl.Errorf("image iter failed. %s %v", id, err)
		return
	}
	return
}

func (s *ImageGroupService) PostImageGroup_Remove(
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
	if err = s._ImageGroupManager.Remove(ctx, uid, id); err != nil {
		xl.Errorf("remove group failed. %s %v", id, err)
	}
	xl.Infof("remove group %s", id)
	return nil
}

func (s *ImageGroupService) PostImageGroup_Delete(
	ctx context.Context,
	req *struct {
		CmdArgs []string
		Images  []string `json:"images"`
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
	ig, err := s._ImageGroupManager.Get(ctx, uid, id)
	if err != nil {
		xl.Errorf("get group failed. %s %v", id, err)
		return err
	}
	if err = ig.Del(ctx, req.Images); err != nil {
		xl.Errorf("del groups failed. %d %v", len(req.Images), err)
		return err
	}
	xl.Infof("group %s delete image", id)
	return nil
}

func (s *ImageGroupService) PostImageGroup_New(
	ctx context.Context,
	req *ImageGroupAddReq,
	env *authstub.Env,
) (ret *ImageGroupAddResp, err error) {
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

	ig, err := s._ImageGroupManager.New(ctx, uid, gid, fv)
	if mgo.IsDup(err) {
		err = httputil.NewError(http.StatusBadRequest, `group already exists`)
		return
	}
	if err != nil {
		return ret, err
	}
	xl.Infof("create group %s", gid)

	ret = s.addImage(ctx, req.Data, ig, api.(ImageGFeatureAPI), uid, utype, gid)
	xl.Infof("group %s add image", gid)
	return
}

func (s *ImageGroupService) PostImageGroup_Add(
	ctx context.Context,
	req *ImageGroupAddReq,
	env *authstub.Env,
) (ret *ImageGroupAddResp, err error) {
	var (
		uid   = env.UserInfo.Uid
		utype = env.UserInfo.Utype
		gid   = req.CmdArgs[0]
	)
	ctx, xl := util.CtxAndLog(ctx, env.W, env.Req)

	_RequestGauge("add").Inc()
	defer func(t1 time.Time) {
		var code = 200
		if err != nil {
			code = httputil.DetectCode(err)
		}
		_ResponseTimeHistogram("add", code).Observe(float64(time.Since(t1) / time.Second))
	}(time.Now())

	ig, err := s._ImageGroupManager.Get(ctx, uid, gid)
	if err != nil {
		xl.Errorf("get group failed. %s %v", gid, err)
		return ret, err
	}

	hid, hub := ig.Hub(ctx)
	_, api, err := s.getFeatureAPI(ctx, hub, hid)
	if err != nil {
		xl.Errorf("get feature api failed. %s %v", gid, err)
		return
	}

	ret = s.addImage(ctx, req.Data, ig, api.(ImageGFeatureAPI), uid, utype, gid)
	xl.Infof("group %s add image", gid)
	return
}

//----------------------------------------------------------------------------//

func (s *ImageGroupService) PostImageGroup_Feature(
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

	hub := s._ImageGroupManager.Hub(ctx)
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

	imagegAPI := api.(ImageGFeatureAPI)
	h := env.W.Header()
	h.Set("Content-Length", strconv.FormatInt(int64(req.ReqBody.To-req.ReqBody.From)*int64(imagegAPI.FeatureLength)*4, 10))
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
		if imagegAPI.ReserveByteOrder && imagegAPI.FeatureByteOrder == FG.BigEndian {
			bs = FG.BigEndianToLittleEndian(bs)
		}
		env.W.Write(bs)
	}
}

func (s *ImageGroupService) postImageGroupSearch(
	ctx context.Context, uid uint32, gids []string, ffs [][]byte, lengths []int, thresholds []float32, limit, blockSize int,
) (ret []ImageGroupsSearchResult, err error) {
	blocks := make([]_IGSearchBlock, 0)
	for i, gid := range gids {
		ig, err := s._ImageGroupManager.Get(ctx, uid, gid)
		if err != nil {
			return nil, errors.Wrapf(err, "_ImageGroupManager.Get %v %v", uid, gid)
		}

		hid, hub := ig.Hub(ctx)
		fbs, err := hub.All(ctx, hid, blockSize)
		if err != nil {
			return nil, errors.Wrapf(err, "hub.All %v %v", hid, blockSize)
		}

		featureStr := base64.StdEncoding.EncodeToString(ffs[i])
		for _, fb := range fbs {
			blocks = append(blocks, _IGSearchBlock{
				FeatureBlock:  fb,
				ImageFeature:  &featureStr,
				FeatureLength: lengths[i],
				Threshold:     thresholds[i],
				_IGSearchGroupHub: _IGSearchGroupHub{
					ImageGroup: ig,
					Gid:        gid,
					Hub:        hub,
					Hid:        hid,
				},
			})
		}
	}

	var (
		xl      = xlog.FromContextSafe(ctx)
		results = make([]_IGSearchResultItem, 0)
		lock    sync.Mutex
		wg      sync.WaitGroup
	)
	for _, block := range blocks {
		wg.Add(1)
		go func(ctx context.Context, sb _IGSearchBlock) {
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
				Features:  *sb.ImageFeature,
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
				xl.Errorf("get image match failed. %v", err)
				return
			}
			if len(sResp) == 0 {
				xl.Errorf("get image match failed. %v", sResp)
				return
			}

			lock.Lock()
			defer lock.Unlock()

			temp := results
			results = make([]_IGSearchResultItem, 0, limit)
			i, j := 0, 0
			for len(results) < limit && i < len(temp) && j < len(sResp[0].Items) {
				if temp[i].Score > sResp[0].Items[j].Score {
					results = append(results, temp[i])
					i++
				} else if temp[i].Score < sResp[0].Items[j].Score {
					results = append(results, _IGSearchResultItem{
						_IGSearchGroupHub: sb._IGSearchGroupHub,
						SearchResultItem:  sResp[0].Items[j],
					})
					j++
				} else {
					results = append(results, temp[i])
					results = append(results, _IGSearchResultItem{
						_IGSearchGroupHub: sb._IGSearchGroupHub,
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
				results = append(results, _IGSearchResultItem{
					_IGSearchGroupHub: sb._IGSearchGroupHub,
					SearchResultItem:  sResp[0].Items[j],
				})
			}

		}(util.SpawnContext(ctx), block)
	}
	wg.Wait()

	for _, _ret := range results {
		fid, err := _ret.Hub.Find(ctx, _ret.Hid, _ret.Version, _ret.Index)
		if err != nil {
			xl.Warnf("hub.Find %s %d %d %s", _ret.Hid, _ret.Version, _ret.Index)
			continue
		}
		item, err := _ret.ImageGroup.Get(ctx, string(fid))
		if err == os.ErrNotExist {
			xl.Warnf("ig.Get %s %d %d %s", _ret.Hid, _ret.Version, _ret.Index, fid)
			continue
		}

		score := _ret.Score
		if score < 0 {
			score = 0
		}
		ret = append(ret, ImageGroupsSearchResult{
			ID:    item.ID,
			Label: item.Label,
			Etag:  item.Etag,
			URI:   item.URI,
			Desc:  item.Desc,
			Score: score,
			Group: _ret.Gid,
		})
	}

	return ret, nil
}

// PostImageGroup_Search ...
func (s *ImageGroupService) PostImageGroup_Search(
	ctx context.Context,
	req *ImageGroupSearchReq,
	env *authstub.Env,
) (ret *ImageGroupSearchResp, err error) {
	ctx, xl := util.CtxAndLog(ctx, env.W, env.Req)

	v2Req := &ImageGroupsSearchReq{
		Data: req.Data,
		Params: ImageGroupsSearchReqParams{
			Groups: []string{req.CmdArgs[0]},
			Limit:  req.Params.Limit,
		},
	}

	v2Ret, err := s.PostImageGroupsSearch(ctx, v2Req, env)
	if err != nil {
		xl.Error("call PostImageGroupsSearch failed. %v", err)
		return
	}

	ret = convertSearchRespFromMultiToSingle(v2Ret)
	return
}

func (s *ImageGroupService) PostImageGroupsSearch(
	ctx context.Context,
	req *ImageGroupsSearchReq,
	env *authstub.Env,
) (ret *ImageGroupsSearchResp, err error) {

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
	}(time.Now())

	if strings.TrimSpace(req.Data.URI) == "" {
		xl.Error("empty data.uri")
		err = httputil.NewError(http.StatusNotAcceptable, "no enough arguments provided")
		return
	}

	if req.Params.Limit > MAX_IMAGE_SEARCH_LIMIT {
		xl.Errorf("invalid params.limit, max allowed: %d\n", MAX_IMAGE_SEARCH_LIMIT)
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
	igs := make([]_ImageGroup, 0)
	for _, id := range groupIds {
		ig, err := s._ImageGroupManager.Get(ctx, uid, id)
		if err != nil {
			xl.Errorf("get group failed. %s %v", id, err)
			return nil, err
		}
		igs = append(igs, ig)
	}

	//get limit
	limit := req.Params.Limit
	// negative limit means want to get all possible satisfied image
	// so set limit to the total number of the group
	if limit < 0 {
		limit = 0
		for _, v := range igs {
			n, err := v.Count(ctx)
			if err != nil {
				xl.Errorf("get image count failed. %v", err)
				return nil, err
			}
			limit += n
			//alway set the limit even return want to all the result
			if limit > MAX_IMAGE_SEARCH_LIMIT {
				limit = MAX_IMAGE_SEARCH_LIMIT
				break
			}
		}
	}
	//if limit is still 0, set it to 1
	if limit <= 0 {
		limit = 1
	}

	//get api & threshold & featureLength of all groups
	imagegAPIs := make([]ImageGFeatureAPI, 0)
	thresholds := make([]float32, 0)
	featureLengths := make([]int, 0)
	for i, v := range igs {
		hid, hub := v.Hub(ctx)
		_, api, err := s.getFeatureAPI(ctx, hub, hid)
		if err != nil {
			xl.Errorf("get feature api failed. %s %v", groupIds[i], err)
			return nil, err
		}
		imageAPI := api.(ImageGFeatureAPI)
		featureLengths = append(featureLengths, imageAPI.FeatureLength)
		imagegAPIs = append(imagegAPIs, imageAPI)

		threshold := req.Params.Threshold
		if threshold < imageAPI.Threshold {
			threshold = imageAPI.Threshold
		}
		thresholds = append(thresholds, threshold)
	}

	ffs, err := s.getImageFeatures(ctx, imagegAPIs, req.Data.URI, uid, utype)
	if err != nil {
		return nil, err
	}

	ret = &ImageGroupsSearchResp{}

	t2 := time.Now()
	ret.Result, err = s.postImageGroupSearch(ctx, uid, groupIds,
		ffs, featureLengths, thresholds, limit, 64*1024/16, // TODO
	)
	if err != nil {
		_ClientTimeHistogram("search", httputil.DetectCode(err)).Observe(float64(time.Since(t2) / time.Second))
	} else {
		_ClientTimeHistogram("search", 200).Observe(float64(time.Since(t2) / time.Second))
	}

	xl.Infof("group %s search image", groupIds)
	if utype != NoChargeUtype {
		util.SetStateHeader(env.W.Header(), "IMAGE_GROUP_SEARCH", 1)
	}
	return
}

func (s *ImageGroupService) getFeatureAPI(ctx context.Context, hub FG.Hub, hid FG.HubID) (FG.FeatureVersion, FG.FeatureAPI, error) {
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

func (s *ImageGroupService) getImageFeatures(ctx context.Context, apis []ImageGFeatureAPI, uri string, uid, utype uint32) (ffs [][]byte, err error) {
	var (
		xl       = xlog.FromContextSafe(ctx)
		waiter   = sync.WaitGroup{}
		lock     = new(sync.Mutex)
		ffMap    = make(map[string][]byte)
		versions []string
		iReq     evals.SimpleReq
	)
	iReq.Data.URI = uri

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
		go func(imageApi ImageGFeatureAPI, ver string) {
			defer waiter.Done()
			ff, err1 := imageApi.IImageFeature.Eval(ctx, iReq, uid, utype)
			if err1 != nil {
				xl.Errorf("get image feature failed. %v", err1)
				lock.Lock()
				if err == nil {
					err = err1
				}
				lock.Unlock()
				return
			}

			// search worker always expects Little-Endian
			if imageApi.ReserveByteOrder && imageApi.FeatureByteOrder == FG.BigEndian {
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

func dataURIEtag(uri string) (string, error) {
	data, err := base64.StdEncoding.DecodeString(strings.TrimPrefix(uri, URI.DataURIPrefix))
	if err != nil {
		return "", err
	}
	h := sha1.New()
	h.Write(data)
	return hex.Dump(h.Sum(nil)), nil
}

func callRetry(ctx context.Context, f func(context.Context) error) error {
	return ahttp.CallWithRetry(ctx, []int{530}, []func(context.Context) error{f, f})
}

func convertSearchRespFromMultiToSingle(multi *ImageGroupsSearchResp) *ImageGroupSearchResp {
	single := &ImageGroupSearchResp{
		Code:    multi.Code,
		Message: multi.Message,
		Result:  make([]ImageGroupSearchResult, 0),
	}
	for _, v := range multi.Result {
		single.Result = append(single.Result, ImageGroupSearchResult{
			ID:    v.ID,
			Label: v.Label,
			Etag:  v.Etag,
			URI:   v.URI,
			Score: v.Score,
			Desc:  v.Desc,
		})
	}
	return single
}
