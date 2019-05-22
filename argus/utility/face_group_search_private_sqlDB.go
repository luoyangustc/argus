package utility

import (
	"bytes"
	"context"
	"database/sql"
	"encoding/base64"
	"encoding/binary"
	"strings"
	"sync"
	"time"

	"github.com/qiniu/xlog.v1"
	"qbox.us/errors"
	"qiniu.com/auth/authstub.v1"
)

type PFaceGroupService struct {
	_FaceGroupManager

	iFaceDetect
	iFaceFeature
	iFaceGroupSearch

	cache map[uint32]map[string]struct {
		Faces    []_FaceItem
		Features string
	}
	*sync.Mutex
}

func NewPFaceGroupServic(c EvalsConfig, manager _FaceGroupManager) *PFaceGroupService {
	s := &PFaceGroupService{
		_FaceGroupManager: manager,
		cache: make(map[uint32]map[string]struct {
			Faces    []_FaceItem
			Features string
		}),
		Mutex: new(sync.Mutex),
	}
	const (
		_CmdFaceDetect      = "facex-detect"
		_CmdFaceFeature     = "facex-feature"
		_CmdFaceFeatureV2   = "facex-feature:v2"
		_CmdFaceGroupSearch = "faceg-search"
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
	return s
}

func (s PFaceGroupService) parseFace(
	ctx context.Context, uri, name string, env _EvalEnv,
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
		Name:    name,
		Feature: ff,
	}, nil

}

func (s *PFaceGroupService) GetFaceGroup(
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

func (s *PFaceGroupService) GetFaceGroup_(
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

func (s *PFaceGroupService) PostFaceGroup_Remove(
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
	func() {
		s.Lock()
		defer s.Unlock()
		if m, ok := s.cache[uid]; ok {
			delete(m, id)
			if len(m) == 0 {
				delete(s.cache, uid)
			}
		}
	}()

	ctx, xl := ctxAndLog(ctx, env.W, env.Req)
	if err := s._FaceGroupManager.Remove(ctx, uid, id); err != nil {
		xl.Errorf("remove group failed. %s %v", id, err)
	}
	return nil
}

func (s *PFaceGroupService) PostFaceGroup_Delete(
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
	func() {
		s.Lock()
		defer s.Unlock()
		if m, ok := s.cache[uid]; ok {
			delete(m, id)
			if len(m) == 0 {
				delete(s.cache, uid)
			}
		}
	}()

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

func (s *PFaceGroupService) PostFaceGroup_New(
	ctx context.Context,
	req *struct {
		CmdArgs []string
		Data    []struct {
			URI       string `json:"uri"`
			Attribute struct {
				Name string `json:"name"`
			} `json:"attribute"`
		} `json:"data"`
	},
	env *authstub.Env,
) (
	ret struct {
		Faces []string `json:"faces,omitempty"`
	},
	err error) {

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
		go func(ctx context.Context, index int, uri, name string) {
			defer waiter.Done()

			item, err := s.parseFace(ctx, uri, name, evalEnv)
			if err != nil {
				return
			}
			lock.Lock()
			defer lock.Unlock()
			faces = append(faces, item)
			indexes[index] = len(faces) - 1

		}(spawnContext(ctx), i, item.URI, item.Attribute.Name)
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

func (s *PFaceGroupService) PostFaceGroup_Add(
	ctx context.Context,
	req *struct {
		CmdArgs []string
		Data    []struct {
			URI       string `json:"uri"`
			Attribute struct {
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
	func() {
		s.Lock()
		defer s.Unlock()
		if m, ok := s.cache[uid]; ok {
			delete(m, id)
			if len(m) == 0 {
				delete(s.cache, uid)
			}
		}
	}()

	ctx, xl := ctxAndLog(ctx, env.W, env.Req)

	fg, err := s._FaceGroupManager.Get(ctx, uid, req.CmdArgs[0])
	if err != nil {
		xl.Errorf("get group failed. %s %v", req.CmdArgs[0], err)
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
		go func(ctx context.Context, index int, uri, name string) {
			defer waiter.Done()

			iface, err := s.parseFace(ctx, uri, name, evalEnv)
			if err != nil {
				return
			}
			lock.Lock()
			defer lock.Unlock()
			faces = append(faces, iface)
			indexes[index] = len(faces) - 1

		}(spawnContext(ctx), i, item.URI, item.Attribute.Name)
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

//----------------------------------------------------------------------------//

type _EvalPFaceGroupSearchReq struct {
	Data []struct {
		URI string `json:"uri"`
	} `json:"data"`
	Params struct {
		Threshold float32 `json:"threshold"`
	} `json:"params"`
}

type _EvalPFaceGroupSearchResp struct {
	Code    int    `json:"code"`
	Message string `json:"message"`
	Result  struct {
		Index int     `json:"index"`
		Score float32 `json:"score"`
	} `json:"result"`
}

type _PFaceGroupSearch struct {
	url     string
	timeout time.Duration
}

func newPFaceGroupSearch(host string, timeout time.Duration) _FaceGroupSearch {
	return _FaceGroupSearch{url: host + "/v1/eval/faceg-search", timeout: timeout}
}

func (fm _PFaceGroupSearch) Eval(
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
func (s *PFaceGroupService) PostFaceGroup_Search(
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
	)
	ctx, xl := ctxAndLog(ctx, env.W, env.Req)

	if strings.TrimSpace(req.Data.URI) == "" {
		xl.Error("empty data.uri")
		return nil, ErrArgs
	}

	items, features, err := func() ([]_FaceItem, string, error) {
		s.Lock()
		if m, ok := s.cache[uid]; ok {
			if ss, ok := m[id]; ok {
				s.Unlock()
				return ss.Faces, ss.Features, nil
			}
		}
		s.Unlock()

		fg, err := s._FaceGroupManager.Get(ctx, uid, req.CmdArgs[0])
		if err != nil {
			xl.Errorf("get group failed. %s %v", req.CmdArgs[0], err)
			return nil, "", err
		}
		items, err := fg.All(ctx)
		if err != nil {
			xl.Errorf("all faces failed. %s %v", req.CmdArgs[0], err)
			return nil, "", err
		}
		var (
			buf     = bytes.NewBuffer(nil)
			base64W = base64.NewEncoder(base64.StdEncoding, buf)

			b4 = make([]byte, 4)
		)
		binary.LittleEndian.PutUint32(b4, uint32(len(items[0].Feature)))
		base64W.Write(b4)
		for _, item := range items {
			base64W.Write(item.Feature)
		}
		base64W.Close()
		features := "data:application/octet-stream;base64," + buf.String()

		s.Lock()
		defer s.Unlock()
		m, ok := s.cache[uid]
		if !ok {
			m = map[string]struct {
				Faces    []_FaceItem
				Features string
			}{}
		}
		m[id] = struct {
			Faces    []_FaceItem
			Features string
		}{
			Faces:    items,
			Features: features,
		}
		s.cache[uid] = m
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

			var mReq = _EvalFaceGroupSearchReq{
				Data: make([]struct {
					URI string `json:"uri"`
				}, 2),
			}
			mReq.Params.Threshold = 0.525
			mReq.Data[0].URI = features
			mReq.Data[1].URI = "data:application/octet-stream;base64," +
				base64.StdEncoding.EncodeToString(ff)
			mResp, err1 := s.iFaceGroupSearch.Eval(ctx, mReq, evalEnv)
			if err1 != nil {
				xl.Errorf("get face match failed. %v", err1)
				lock.Lock()
				if err == nil {
					err = err1
				}
				lock.Unlock()
				return
			}

			if mResp.Result.Index < 0 || mResp.Result.Score < 0.01 {
				return
			}
			detail := FaceSearchDetail{
				BoundingBox: FaceDetectBox{
					Pts:   face.Pts,
					Score: face.Score,
				},
			}
			detail.Value.Name = items[mResp.Result.Index].Name
			detail.Value.Score = mResp.Result.Score
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

type pfaceGroupManagerInDB struct {
	db *sql.DB
}

func NewPFaceGroupManagerInDB(db *sql.DB) (_FaceGroupManager, error) {

	db.Exec(
		"CREATE TABLE groups (" +
			"uid INT," +
			"id CHARACTER(32)," +
			"PRIMARY KEY (uid, id)" +
			")",
	)

	db.Exec(
		"CREATE TABLE faces (" +
			"uid INT," +
			"gid CHARACTER(32)," +
			"id CHARACTER(32)," +
			"name CHARACTER(128)," +
			"feature BLOB," +
			"PRIMARY KEY (uid, gid, id)" +
			")",
	)

	return pfaceGroupManagerInDB{db: db}, nil
}

func (m pfaceGroupManagerInDB) Get(ctx context.Context, uid uint32, id string) (_FaceGroup, error) {

	row := m.db.QueryRowContext(ctx, "SELECT id FROM groups WHERE uid = $1 AND id = $2", uid, id)
	var _id string
	if err := row.Scan(&_id); err != nil {
		return nil, err
	}
	return pfaceGroupInDB{db: m.db, uid: uid, id: id}, nil
}

func (m pfaceGroupManagerInDB) All(ctx context.Context, uid uint32) ([]string, error) {
	var ids = make([]string, 0)
	rows, err := m.db.QueryContext(ctx, "SELECT id FROM groups WHERE uid = $1", uid)
	if err != nil {
		return nil, err
	}
	for rows.Next() {
		var id string
		if err := rows.Scan(&id); err != nil {
			return nil, err
		}
		ids = append(ids, id)
	}
	if err := rows.Err(); err != nil {
		return nil, err
	}
	return ids, nil
}

func (m pfaceGroupManagerInDB) New(ctx context.Context, uid uint32, id string) (_FaceGroup, error) {
	_, err := m.db.ExecContext(ctx, "INSERT INTO groups (uid, id) VALUES ($1, $2)", uid, id)
	if err != nil {
		return nil, err
	}
	return pfaceGroupInDB{db: m.db, uid: uid, id: id}, nil
}

func (m pfaceGroupManagerInDB) Remove(ctx context.Context, uid uint32, id string) error {
	m.db.ExecContext(ctx, "DELETE FROM faces WHERE uid = $1 AND gid = $2", uid, id)
	m.db.ExecContext(ctx, "DELETE FROM groups WHERE uid = $1 AND id = $2", uid, id)
	return nil
}

type pfaceGroupInDB struct {
	db  *sql.DB
	uid uint32
	id  string
}

func (g pfaceGroupInDB) Add(ctx context.Context, items []_FaceItem) ([]string, error) {
	xl := xlog.FromContextSafe(ctx)
	ids := make([]string, 0, len(items))
	for _, item := range items {
		if len(item.ID) == 0 {
			item.ID = xlog.GenReqId()
		}
		_, err := g.db.ExecContext(ctx,
			"INSERT INTO faces (uid, gid, id, name, feature) "+
				"VALUES ($1, $2, $3, $4, $5)",
			g.uid, g.id, item.ID, item.Name, item.Feature,
		)
		xl.Infof("insert face. %v", err)
		ids = append(ids, item.ID)
	}
	return ids, nil
}

func (g pfaceGroupInDB) Del(ctx context.Context, ids []string) error {
	xl := xlog.FromContextSafe(ctx)
	for _, id := range ids {
		if _, err := g.db.ExecContext(ctx,
			"DELETE FROM faces WHERE uid = $1 AND gid = $2 AND id = $3",
			g.uid, g.id, id,
		); err != nil {
			xl.Warnf("delete face failed. %s %v", id, err)
		}
	}
	return nil
}

func (g pfaceGroupInDB) All(ctx context.Context) ([]_FaceItem, error) {
	rows, err := g.db.QueryContext(ctx,
		"SELECT id, name, feature FROM faces WHERE uid = $1 AND gid = $2",
		g.uid, g.id,
	)
	if err != nil {
		return nil, err
	}
	items := make([]_FaceItem, 0)
	for rows.Next() {
		var (
			id      string
			name    string
			feature []byte
		)
		if err := rows.Scan(&id, &name, &feature); err != nil {
			return nil, err
		}
		items = append(items,
			_FaceItem{
				ID:      id,
				Name:    name,
				Feature: feature,
			})
	}
	if err := rows.Err(); err != nil {
		return nil, err
	}
	return items, nil
}

func (g pfaceGroupInDB) CheckByID(ctx context.Context, id string) (bool, error) {
	// empty function
	return true, nil
}
