package utility

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/binary"
	"fmt"
	"strings"
	"sync"
	"time"

	"gopkg.in/mgo.v2"

	"github.com/qiniu/db/mgoutil.v3"
	"github.com/qiniu/xlog.v1"
	"gopkg.in/mgo.v2/bson"
	"qbox.us/errors"
	"qiniu.com/auth/authstub.v1"
)

type FaceGroupService struct {
	_FaceGroupManager

	iFaceDetect
	iFaceFeature
	iFaceGroupSearch
	iHhFaceSearchStatic

	// cache map[uint32]map[string]struct {
	// 	Faces    []_FaceItem
	// 	Features string
	// }
	// *sync.Mutex
}

func NewFaceGroupService() *FaceGroupService { return &FaceGroupService{} }
func (s FaceGroupService) Version() string   { return "20171210" }
func (s FaceGroupService) Config() interface{} {
	return struct {
		Evals EvalsConfig     `json:"evals"`
		Mgo   *mgoutil.Config `json:"mgo"`
	}{}
}

func (s *FaceGroupService) Init(v interface{}) error {

	conf := v.(struct {
		Evals EvalsConfig     `json:"evals"`
		Mgo   *mgoutil.Config `json:"mgo"`
	})

	s._FaceGroupManager, _ = NewFaceGroupManagerInDB(conf.Mgo)
	_ = s.init(conf.Evals)
	return nil
}

func NewFaceGroupServic(c EvalsConfig, manager _FaceGroupManager) *FaceGroupService {
	s := &FaceGroupService{
		_FaceGroupManager: manager,
		// cache: make(map[uint32]map[string]struct {
		// 	Faces    []_FaceItem
		// 	Features string
		// }),
		// Mutex: new(sync.Mutex),
	}
	s.init(c)
	return s
}

func (s *FaceGroupService) init(c EvalsConfig) error {
	const (
		_CmdFaceDetect         = "facex-detect"
		_CmdFaceFeature        = "facex-feature"
		_CmdFaceFeatureV2      = "facex-feature:v2"
		_CmdFaceGroupSearch    = "faceg-search"
		_CmdHhFaceSearchStatic = "hh-search-static"
	)
	{
		conf := c.Get(_CmdFaceDetect)
		fd := newFaceDetect(conf.Host, conf.Timeout)
		if conf.URL != "" {
			fd.url = conf.URL
		}
		if conf.Auth.AK != "" {
			fd.Client = newQiniuAuthClient(conf.Auth.AK, conf.Auth.SK, conf.Timeout)
		}
		s.iFaceDetect = fd
	}
	{
		var ff _FaceFeature
		if _, ok := c.Evals[_CmdFaceFeatureV2]; ok {
			conf := c.Get(_CmdFaceFeatureV2)
			ff = newFaceFeature(conf.Host, conf.Timeout)
			ff.url = conf.Host + "/v1/eval/facex-feature-v2"
			if conf.URL != "" {
				ff.url = conf.URL
			}
			if conf.Auth.AK != "" {
				ff.Client = newQiniuAuthClient(conf.Auth.AK, conf.Auth.SK, conf.Timeout)
			}
		} else {
			conf := c.Get(_CmdFaceFeature)
			ff = newFaceFeature(conf.Host, conf.Timeout)
			if conf.URL != "" {
				ff.url = conf.URL
			}
			if conf.Auth.AK != "" {
				ff.Client = newQiniuAuthClient(conf.Auth.AK, conf.Auth.SK, conf.Timeout)
			}
		}
		s.iFaceFeature = ff
	}
	{
		conf := c.Get(_CmdFaceGroupSearch)
		g := newFaceGroupSearch(conf.Host, conf.Timeout)
		if conf.URL != "" {
			g.url = conf.URL
		}
		s.iFaceGroupSearch = g
	}
	{
		conf := c.Get(_CmdHhFaceSearchStatic)
		g := newHhFaceSearchStatic(conf.Host, conf.Timeout)
		s.iHhFaceSearchStatic = g
	}
	return nil
}

func (s FaceGroupService) parseFace(
	ctx context.Context, uri string, env _EvalEnv,
) (_FaceItem, error) {

	var (
		xl   = xlog.FromContextSafe(ctx)
		item _FaceItem
	)

	dResp, err := s.iFaceDetect.Eval(ctx,
		_EvalFaceDetectReq{Data: struct {
			URI string `json:"uri"`
		}{URI: uri}},
		env)
	if err != nil {
		xl.Errorf("call facex-detect error: %s %v", uri, err)
		return item, err
	}

	xl.Infof("face detect: %#v", dResp.Result)

	if len(dResp.Result.Detections) != 1 {
		xl.Warnf("not one face: %s %d", uri, len(dResp.Result.Detections))
		return item, errors.New("not face detected")
	}

	one := dResp.Result.Detections[0]
	var fReq _EvalFaceReq
	fReq.Data.URI = uri
	fReq.Data.Attribute.Pts = one.Pts
	ff, err := s.iFaceFeature.Eval(ctx, fReq, env)
	if err != nil {
		xl.Errorf("get face feature failed. %s %v", uri, err)
		return item, err
	}

	xl.Infof("face feature: %d", len(ff))

	return _FaceItem{
		Feature: ff,
	}, nil

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
	ctx, xl := ctxAndLog(ctx, env.W, env.Req)

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
	ctx, xl := ctxAndLog(ctx, env.W, env.Req)

	fg, err := s._FaceGroupManager.Get(ctx, uid, id)
	if err != nil {
		xl.Errorf("get group failed. %s %v", id, err)
		return
	}
	items, err := fg.All(ctx)
	if err != nil {
		xl.Errorf("all face failed. %s %v", id, err)
		return
	}
	xl.Infof("all face done. %d", len(items))

	ret.Result = make([]struct {
		ID    string `json:"id"`
		Value struct {
			Name string `json:"name"`
		} `json:"value"`
	}, 0, len(items))
	for _, item := range items {
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
	// func() {
	// 	s.Lock()
	// 	defer s.Unlock()
	// 	if m, ok := s.cache[uid]; ok {
	// 		delete(m, id)
	// 		if len(m) == 0 {
	// 			delete(s.cache, uid)
	// 		}
	// 	}
	// }()

	ctx, xl := ctxAndLog(ctx, env.W, env.Req)
	if err := s._FaceGroupManager.Remove(ctx, uid, id); err != nil {
		xl.Errorf("remove group failed. %s %v", id, err)
	}
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
	// func() {
	// 	s.Lock()
	// 	defer s.Unlock()
	// 	if m, ok := s.cache[uid]; ok {
	// 		delete(m, id)
	// 		if len(m) == 0 {
	// 			delete(s.cache, uid)
	// 		}
	// 	}
	// }()

	ctx, xl := ctxAndLog(ctx, env.W, env.Req)
	fg, err := s._FaceGroupManager.Get(ctx, uid, id)
	if err != nil {
		xl.Errorf("get group failed. %s %v", id, err)
		return err
	}
	if err := fg.Del(ctx, req.Faces); err != nil {
		xl.Errorf("del faces failed. %d %v", len(req.Faces), err)
		return err
	}
	return nil
}

func (s *FaceGroupService) PostFaceGroup_New(
	ctx context.Context,
	req *struct {
		CmdArgs []string
		Data    []struct {
			URI       string `json:"uri"`
			Attribute struct {
				ID   string `json:"id"`
				Name string `json:"name"`
			} `json:"attribute"`
		} `json:"data"`
	},
	env *authstub.Env,
) (
	ret struct {
		Faces []string `json:"faces,omitempty"`
	}, err error) {

	var (
		uid     = env.UserInfo.Uid
		utype   = env.UserInfo.Utype
		evalEnv = _EvalEnv{Uid: uid, Utype: utype}
	)
	ctx, xl := ctxAndLog(ctx, env.W, env.Req)

	fg, err := s._FaceGroupManager.New(ctx, uid, req.CmdArgs[0])
	if err != nil {
		return
	}

	var (
		waiter = sync.WaitGroup{}
		lock   = new(sync.Mutex)
		faces  = make([]_FaceItem, 0, len(req.Data))

		indexes = make(map[int]int)
	)

	for i, item := range req.Data {
		waiter.Add(1)
		go func(ctx context.Context, index int, uri, id, name string) {
			defer waiter.Done()

			item, err := s.parseFace(ctx, uri, evalEnv)
			if err != nil {
				return
			}
			item.ID = id
			item.Name = name
			lock.Lock()
			defer lock.Unlock()
			faces = append(faces, item)
			indexes[index] = len(faces) - 1

		}(spawnContext(ctx), i,
			item.URI, item.Attribute.ID, item.Attribute.Name)
	}

	waiter.Wait()
	ret.Faces = make([]string, len(faces))
	if ids, err := fg.Add(ctx, faces); err != nil {
		xl.Errorf("ad face failed. %d %v", len(faces), err)
	} else {
		for i, j := range indexes {
			ret.Faces[i] = ids[j]
		}
	}

	return
}

func (s *FaceGroupService) PostFaceGroup_Add(
	ctx context.Context,
	req *struct {
		CmdArgs []string
		Data    []struct {
			URI       string `json:"uri"`
			Attribute struct {
				ID   string `json:"id"`
				Name string `json:"name"`
			} `json:"attribute"`
		} `json:"data"`
	},
	env *authstub.Env,
) (
	ret struct {
		Faces []string `json:"faces,omitempty"`
	}, err error) {

	var (
		uid     = env.UserInfo.Uid
		utype   = env.UserInfo.Utype
		evalEnv = _EvalEnv{Uid: uid, Utype: utype}
		id      = req.CmdArgs[0]
	)
	// func() {
	// 	s.Lock()
	// 	defer s.Unlock()
	// 	if m, ok := s.cache[uid]; ok {
	// 		delete(m, id)
	// 		if len(m) == 0 {
	// 			delete(s.cache, uid)
	// 		}
	// 	}
	// }()

	ctx, xl := ctxAndLog(ctx, env.W, env.Req)

	fg, err := s._FaceGroupManager.Get(ctx, uid, id)
	if err != nil {
		xl.Errorf("get group failed. %s %v", id, err)
		return
	}

	var (
		waiter = sync.WaitGroup{}
		lock   = new(sync.Mutex)
		faces  = make([]_FaceItem, 0, len(req.Data))

		indexes = make(map[int]int)
	)

	for i, item := range req.Data {
		waiter.Add(1)
		go func(ctx context.Context, index int, uri, id, name string) {
			defer waiter.Done()

			iface, err := s.parseFace(ctx, uri, evalEnv)
			if err != nil {
				return
			}
			iface.ID = id
			iface.Name = name
			lock.Lock()
			defer lock.Unlock()
			faces = append(faces, iface)
			indexes[index] = len(faces) - 1

		}(spawnContext(ctx), i,
			item.URI, item.Attribute.ID, item.Attribute.Name)
	}

	waiter.Wait()
	ret.Faces = make([]string, len(req.Data))
	if ids, err := fg.Add(ctx, faces); err != nil {
		xl.Errorf("ad face failed. %d %v", len(faces), err)
	} else {
		for i := 0; i < len(req.Data); i++ {
			if j, ok := indexes[i]; !ok {
				ret.Faces[i] = ""
			} else {
				ret.Faces[i] = ids[j]
			}
		}
	}

	return
}

//----------------------------------------------------------------------------//

type _EvalFaceGroupSearchReq struct {
	Data []struct {
		URI string `json:"uri"`
	} `json:"data"`
	Params struct {
		Threshold float32 `json:"threshold"`
	} `json:"params"`
}

type _EvalFaceGroupSearchResp struct {
	Code    int    `json:"code"`
	Message string `json:"message"`
	Result  struct {
		Index int     `json:"index"`
		Score float32 `json:"score"`
	} `json:"result"`
}

type iFaceGroupSearch interface {
	Eval(context.Context, _EvalFaceGroupSearchReq, _EvalEnv) (_EvalFaceGroupSearchResp, error)
}

type _FaceGroupSearch struct {
	url     string
	timeout time.Duration
}

func newFaceGroupSearch(host string, timeout time.Duration) _FaceGroupSearch {
	return _FaceGroupSearch{url: host + "/v1/eval/facex-search", timeout: timeout}
}

func (fm _FaceGroupSearch) Eval(
	ctx context.Context, req _EvalFaceGroupSearchReq, env _EvalEnv,
) (_EvalFaceGroupSearchResp, error) {
	var (
		client = newRPCClient(env, fm.timeout)

		resp _EvalFaceGroupSearchResp
	)
	err := client.CallWithJson(ctx, &resp, "POST", fm.url, &req)
	return resp, err
}

//----------------------------------------------------------------------------//

// PostFaceGroup_Search ...
func (s *FaceGroupService) PostFaceGroup_Search(
	ctx context.Context,
	req *struct {
		CmdArgs []string
		Data    struct {
			URI string `json:"uri"`
		} `json:"data"`
	},
	env *authstub.Env,
) (ret *FaceSearchResp, err error) {

	var (
		uid     = env.UserInfo.Uid
		utype   = env.UserInfo.Utype
		evalEnv = _EvalEnv{Uid: uid, Utype: utype}
		id      = req.CmdArgs[0]
		fg      _FaceGroup
	)
	ctx, xl := ctxAndLog(ctx, env.W, env.Req)

	if strings.TrimSpace(req.Data.URI) == "" {
		xl.Error("empty data.uri")
		return nil, ErrArgs
	}

	items, features, err := func() ([]_FaceItem, string, error) {
		// s.Lock()
		// if m, ok := s.cache[uid]; ok {
		// 	if ss, ok := m[id]; ok {
		// 		s.Unlock()
		// 		return ss.Faces, ss.Features, nil
		// 	}
		// }
		// s.Unlock()

		fg, err = s._FaceGroupManager.Get(ctx, uid, req.CmdArgs[0])
		if err != nil {
			xl.Errorf("get group failed. %s %v", req.CmdArgs[0], err)
			return nil, "", err
		}
		items, err := fg.All(ctx)
		if err != nil {
			xl.Errorf("all faces failed. %s %v", req.CmdArgs[0], err)
			return nil, "", err
		}
		if len(items) == 0 {
			return nil, "", nil
		}
		var (
			buf     = bytes.NewBuffer(nil)
			base64W = base64.NewEncoder(base64.StdEncoding, buf)

			b4 = make([]byte, 4)
		)
		if len(items) > 0 {
			binary.LittleEndian.PutUint32(b4, uint32(len(items[0].Feature)))
			base64W.Write(b4)
		}
		for _, item := range items {
			base64W.Write(item.Feature)
		}
		base64W.Close()
		features := "data:application/octet-stream;base64," + buf.String()

		// s.Lock()
		// defer s.Unlock()
		// m, ok := s.cache[uid]
		// if !ok {
		// 	m = map[string]struct {
		// 		Faces    []_FaceItem
		// 		Features string
		// 	}{}
		// }
		// m[id] = struct {
		// 	Faces    []_FaceItem
		// 	Features string
		// }{
		// 	Faces:    items,
		// 	Features: features,
		// }
		// s.cache[uid] = m
		return items, features, nil
	}()
	if err != nil {
		return nil, err
	}

	dResp, err := s.iFaceDetect.Eval(ctx,
		_EvalFaceDetectReq{Data: struct {
			URI string `json:"uri"`
		}{URI: req.Data.URI}},
		evalEnv)
	if err != nil {
		xl.Errorf("call facex-detect error: %s %v", req.Data.URI, err)
		return nil, err
	}
	if dResp.Code != 0 && dResp.Code/100 != 2 {
		xl.Errorf("call facex-detect failed: %s %d %s", req.Data.URI, dResp.Code, dResp.Message)
		return nil, errors.New("call facex-detect failed")
	}

	var (
		waiter = sync.WaitGroup{}
		lock   = new(sync.Mutex)
	)
	ret = &FaceSearchResp{
		Result: FaceSearchResult{
			Detections: make([]FaceSearchDetail, 0, len(dResp.Result.Detections)),
		},
	}
	ctxs := make([]context.Context, 0, len(dResp.Result.Detections))
	for _, d := range dResp.Result.Detections {
		waiter.Add(1)
		ctx2 := spawnContext(ctx)
		go func(ctx context.Context, face _EvalFaceDetection) {
			defer waiter.Done()
			xl := xlog.FromContextSafe(ctx)

			var fReq _EvalFaceReq
			fReq.Data.URI = req.Data.URI
			fReq.Data.Attribute.Pts = face.Pts
			ff, err1 := s.iFaceFeature.Eval(ctx, fReq, evalEnv)
			if err1 != nil {
				xl.Errorf("get face feature failed. %v", err1)
				lock.Lock()
				if err == nil {
					err = err1
				}
				lock.Unlock()
				return
			}

			var (
				mResp     _EvalFaceGroupSearchResp
				pResp     _EvalFaceSearchResp
				_inWaiter sync.WaitGroup
				err2      error
			)
			mResp.Result.Index = -1
			pResp.Result.Index = -1
			if len(items) > 0 {
				var mReq = _EvalFaceGroupSearchReq{
					Data: make([]struct {
						URI string `json:"uri"`
					}, 2),
				}
				mReq.Params.Threshold = 0.525
				mReq.Data[0].URI = features
				mReq.Data[1].URI = "data:application/octet-stream;base64," +
					base64.StdEncoding.EncodeToString(ff)
				_inWaiter.Add(1)
				go func(ctx context.Context) {
					defer _inWaiter.Done()
					mResp, err1 = s.iFaceGroupSearch.Eval(ctx, mReq, evalEnv)
				}(spawnContext(ctx))
				if err1 != nil {
					xl.Errorf("get face match failed. %v", err1)
					lock.Lock()
					if err == nil {
						err = err1
					}
					lock.Unlock()
					return
				}
			}

			if id == HahuiGroupID {
				var pReq = _EvalFaceSearchReq{
					Data: struct {
						URI string `json:"uri"`
					}{URI: "data:application/octet-stream;base64," +
						base64.StdEncoding.EncodeToString(ff)},
				}
				_inWaiter.Add(1)
				go func(ctx context.Context) { //statistic library
					defer _inWaiter.Done()
					pResp, err2 = s.iHhFaceSearchStatic.Eval(ctx, pReq, evalEnv)
				}(spawnContext(ctx))
			}
			_inWaiter.Wait()

			if err1 != nil {
				xl.Errorf("get face match failed. %v", err1)
			} else if err2 != nil {
				xl.Errorf("PostFaceGroup_Search hh-search-static error: %v", err1)
			}
			if err1 != nil && err2 != nil {
				lock.Lock()
				if err == nil {
					err = fmt.Errorf("face-search error: %v; hh-search-static: %v", err1, err2)
				}
				lock.Unlock()
				return
			}

			if (mResp.Result.Index < 0 || mResp.Result.Score < 0.01) && (pResp.Result.Index < 0 || pResp.Result.Score < 0.01) {
				return
			}
			detail := FaceSearchDetail{
				BoundingBox: FaceDetectBox{
					Pts:   face.Pts,
					Score: face.Score,
				},
			}

			if mResp.Result.Score > pResp.Result.Score {
				detail.Value.Name = items[mResp.Result.Index].Name
				detail.Value.Score = mResp.Result.Score
			} else {
				found, err1 := fg.CheckByID(spawnContext(ctx), pResp.Result.Sample.ID)
				if err1 != nil {
					lock.Lock()
					if err == nil {
						err = err1
					}
					lock.Lock()
					return
				}
				if pResp.Result.Class != "" && found {
					detail.Value.Name = pResp.Result.Class
					detail.Value.Score = pResp.Result.Score
				}
			}
			lock.Lock()
			defer lock.Unlock()
			ret.Result.Detections = append(ret.Result.Detections, detail)
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

	if len(ret.Result.Detections) == 0 {
		ret.Message = "No valid face info detected"
	}

	setStateHeader(env.W.Header(), "FACE_SEARCH_POLITICIAN", 1)
	return
}

////////////////////////////////////////////////////////////////////////////////

type _FaceGroupManager interface {
	Get(context.Context, uint32, string) (_FaceGroup, error)
	All(context.Context, uint32) ([]string, error)
	New(context.Context, uint32, string) (_FaceGroup, error)
	Remove(context.Context, uint32, string) error
}

type _FaceItem struct {
	ID      string `json:"id" bson:"id"`
	Name    string `json:"name" bson:"name"`
	Feature []byte `json:"feature" bson:"feature"`
}

type _FaceGroup interface {
	Add(context.Context, []_FaceItem) ([]string, error)
	Del(context.Context, []string) error
	All(context.Context) ([]_FaceItem, error)
	CheckByID(context.Context, string) (bool, error)
}

type faceGroupManagerColls struct {
	Groups mgoutil.Collection `coll:"fdbgroups"`
	Faces  mgoutil.Collection `coll:"fdbfaces"`
}

type faceGroupManagerInDB struct {
	groups *mgoutil.Collection
	faces  *mgoutil.Collection
}

func NewFaceGroupManagerInDB(mgoConf *mgoutil.Config) (_FaceGroupManager, error) {

	var (
		colls faceGroupManagerColls
	)
	sess, err := mgoutil.Open(&colls, mgoConf)
	if err != nil {
		return nil, err
	}
	sess.SetPoolLimit(DefaultCollSessionPoolLimit)
	colls.Groups.EnsureIndex(mgo.Index{Key: []string{"uid", "id"}, Unique: true})
	colls.Faces.EnsureIndex(mgo.Index{Key: []string{"uid", "gid", "id"}, Unique: true})
	colls.Faces.EnsureIndex(mgo.Index{Key: []string{"uid", "gid"}})

	return faceGroupManagerInDB{groups: &colls.Groups, faces: &colls.Faces}, nil
}

func (m faceGroupManagerInDB) Get(ctx context.Context, uid uint32, id string) (_FaceGroup, error) {

	g := m.groups.CopySession()
	defer g.CloseSession()

	type Id struct {
		ID string `bson:"id"`
	}
	var _id Id
	err := g.Find(bson.M{"uid": uid, "id": id}).Select(bson.M{"_id": 0, "uid": 0}).One(&_id)
	if err != nil {
		return nil, err
	}
	return faceGroupInDB{faces: m.faces, uid: uid, id: id}, nil
}

func (m faceGroupManagerInDB) All(ctx context.Context, uid uint32) ([]string, error) {

	g := m.groups.CopySession()
	defer g.CloseSession()

	type Id struct {
		ID string `bson:"id"`
	}
	var ids = make([]Id, 0)
	var ret = make([]string, 0)

	err := g.Find(bson.M{"uid": uid}).Select(bson.M{"id": 1}).All(&ids)
	if err != nil {
		return nil, err
	}
	for _, id := range ids {
		ret = append(ret, id.ID)
	}
	return ret, nil
}

func (m faceGroupManagerInDB) New(ctx context.Context, uid uint32, id string) (_FaceGroup, error) {

	g := m.groups.CopySession()
	defer g.CloseSession()

	err := g.Insert(bson.M{"uid": uid, "id": id})
	if err != nil && !mgo.IsDup(err) {
		return nil, err
	}
	return faceGroupInDB{faces: m.faces, uid: uid, id: id}, nil
}

func (m faceGroupManagerInDB) Remove(ctx context.Context, uid uint32, id string) error {

	g := m.groups.CopySession()
	defer g.CloseSession()

	f := m.faces.CopySession()
	defer f.CloseSession()

	xl := xlog.FromContextSafe(ctx)
	_, err := f.RemoveAll(bson.M{"uid": uid, "gid": id})
	if err != nil {
		xl.Errorf("remove those faces with uid %v gid %v error:%v ", uid, id, err)
		return err
	}
	_, err = g.RemoveAll(bson.M{"uid": uid, "id": id})
	if err != nil {
		xl.Errorf("remove group with uid %v gid %v error:%v ", uid, id, err)
		return err
	}
	return nil
}

type faceGroupInDB struct {
	faces *mgoutil.Collection
	uid   uint32
	id    string
}

func (g faceGroupInDB) Add(ctx context.Context, items []_FaceItem) ([]string, error) {

	f := g.faces.CopySession()
	defer f.CloseSession()

	type FaceRecord struct {
		Uid     uint32 `bson:"uid"`
		Gid     string `bson:"gid"`
		ID      string `bson:"id"`
		Name    string `bson:"name"`
		Feature []byte `bson:"feature"`
		Static  bool   `bson:"static"`
	}
	Fitems := make([]interface{}, 0, len(items))
	ids := make([]string, 0, len(items))
	for _, item := range items {
		id := item.ID
		if len(id) == 0 {
			id = xlog.GenReqId()
		}
		Fitems = append(Fitems, FaceRecord{
			Uid:     g.uid,
			Gid:     g.id,
			ID:      id,
			Name:    item.Name,
			Feature: item.Feature,
			Static:  false,
		})
		ids = append(ids, id)
	}

	err := f.Insert(Fitems...)
	return ids, err
}

func (g faceGroupInDB) Del(ctx context.Context, ids []string) error {

	f := g.faces.CopySession()
	defer f.CloseSession()

	_, err := f.RemoveAll(bson.M{"uid": g.uid, "gid": g.id, "id": bson.M{"$in": ids}})
	return err
}

func (g faceGroupInDB) All(ctx context.Context) ([]_FaceItem, error) {

	f := g.faces.CopySession()
	defer f.CloseSession()

	items := make([]_FaceItem, 0)

	err := f.Find(bson.M{"uid": g.uid, "gid": g.id, "$or": []bson.M{bson.M{"static": false}, bson.M{"static": bson.M{"$exists": false}}}}).Select(bson.M{"_id": 0, "uid": 0, "gid": 0}).All(&items)
	return items, err
}

func (g faceGroupInDB) CheckByID(ctx context.Context, id string) (bool, error) {
	f := g.faces.CopySession()
	defer f.CloseSession()

	n, err := f.Find(bson.M{"id": id}).Count()
	if err != nil {
		return false, err
	}
	return n >= 1, nil
}
