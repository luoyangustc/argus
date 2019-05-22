// +build cublas

package feature_search

import (
	"context"
	"net/http"
	"strconv"
	"sync"
	"sync/atomic"
	"time"

	"github.com/qiniu/xlog.v1"
	"qbox.us/net/httputil"
	feature_group "qiniu.com/argus/feature_group_private"
	"qiniu.com/argus/feature_group_private/proto"
	"qiniu.com/argus/feature_group_private/search"
	"qiniu.com/argus/feature_group_private/search/gpu"
	"qiniu.com/auth/authstub.v1"
)

type Config struct {
	search.Config
}

const (
	maxLimit = 1000
)

type Service struct {
	Config
	Sets     search.Sets
	Recycles map[search.SetName]struct {
		Timestamps []struct {
			ID        proto.FeatureID
			Timestamp time.Time
		}
		Lifetime int
	}
	Mutex sync.Mutex
}

func New(c Config) (*Service, error) {
	var err error

	srv := &Service{
		Config: c,
		Recycles: make(map[search.SetName]struct {
			Timestamps []struct {
				ID        proto.FeatureID
				Timestamp time.Time
			}
			Lifetime int
		}, 0),
	}

	srv.Sets, err = gpu.NewSets(c.Config)
	if err != nil {
		return nil, err
	}
	return srv, nil
}

func ctxAndLog(
	ctx context.Context, w http.ResponseWriter, req *http.Request,
) (context.Context, *xlog.Logger) {

	xl, ok := xlog.FromContext(ctx)
	if !ok {
		xl = xlog.New(w, req)
		ctx = xlog.NewContext(ctx, xl)
	}
	return ctx, xl
}

//---------------------------------------------------------------------------------//

func (s *Service) recycle(name search.SetName, set search.Set) {
	for {
		time.Sleep(10 * time.Second)
		s.Mutex.Lock()
		recycle, _ := s.Recycles[name]
		if recycle.Lifetime == 0 {
			s.Mutex.Unlock()
			return
		}
		var remain []struct {
			ID        proto.FeatureID
			Timestamp time.Time
		}

		for _, tm := range recycle.Timestamps {
			if string(tm.ID) != "" && tm.Timestamp.Add(time.Duration(recycle.Lifetime)*time.Second).Before(time.Now()) {
				ctx := context.Background()
				set.Delete(ctx, tm.ID)
				xlog.FromContextSafe(ctx).Infof("Set %s Feature %s out of lifetime, deleted...", name, tm.ID)
				continue
			}
			remain = append(remain, tm)
		}

		recycle.Timestamps = remain
		s.Recycles[name] = recycle
		s.Mutex.Unlock()
	}
}

type PostSets_Req struct {
	CmdArgs   []string
	Dimension int             `json:"dimension"`
	Precision int             `json:"precision"`
	Size      int             `json:"size"`
	Version   uint64          `json:"version"`
	State     search.SetState `json:"state"`
	Timeout   int             `json:"timeout"`
}

func (s *Service) PostSets_(ctx context.Context, req *PostSets_Req, env *authstub.Env) (err error) {

	var (
		name = search.SetName(req.CmdArgs[0])
		set  search.Set
	)
	_, xl := ctxAndLog(ctx, env.W, env.Req)

	if _, err = s.Sets.Get(ctx, name); err != nil && err != search.ErrFeatureSetNotFound {
		return
	}
	if err == nil {
		err = httputil.NewError(http.StatusBadRequest, ErrFeatureSetExist.Error())
		return
	}

	if req.Precision == 0 {
		req.Precision = s.Config.Precision
	}

	if req.Dimension == 0 {
		req.Dimension = s.Config.Dimension
	}

	if err = s.Sets.New(ctx, name, search.Config{
		Dimension: req.Dimension,
		Precision: req.Precision,
		BatchSize: s.BatchSize,
		Version:   req.Version,
	}, req.State); err != nil {
		err = httputil.NewError(http.StatusBadRequest, err.Error())
		return
	}

	if set, err = s.Sets.Get(ctx, name); err != nil {
		return
	}

	if req.Timeout > 0 {
		recycle := struct {
			Timestamps []struct {
				ID        proto.FeatureID
				Timestamp time.Time
			}
			Lifetime int
		}{
			Timestamps: make([]struct {
				ID        proto.FeatureID
				Timestamp time.Time
			}, 0),
			Lifetime: req.Timeout,
		}
		s.Mutex.Lock()
		s.Recycles[name] = recycle
		s.Mutex.Unlock()

		go s.recycle(name, set)
	}
	xl.Debug("Create feature set", name)
	return
}

//---------------------------------------------------------------------------------//
type GetSets_Req struct {
	CmdArgs []string
}

type GetSets_Resp struct {
	Dimension int    `json:"dimension"`
	Precision int    `json:"precision"`
	Size      int    `json:"size"`
	Version   uint64 `json:"version"`
	State     int    `json:"state"`
}

func (s *Service) GetSets_(ctx context.Context, req *GetSets_Req, env *authstub.Env) (resp GetSets_Resp, err error) {
	var (
		name = search.SetName(req.CmdArgs[0])
		set  search.Set
	)
	if set, err = s.Sets.Get(ctx, name); err != nil {
		return
	}

	cfg := set.Config(ctx)
	resp.Dimension = cfg.Dimension
	resp.Precision = cfg.Precision
	resp.Size = cfg.BlockNum * cfg.BlockSize
	resp.Version = atomic.LoadUint64(&cfg.Version)
	return
}

//---------------------------------------------------------------------------------//
type PostSets_State_ struct {
	CmdArgs []string
}

func (s *Service) PostSets_State_(ctx context.Context, req *PostSets_State_, env *authstub.Env) (err error) {
	var (
		name     = search.SetName(req.CmdArgs[0])
		state, _ = strconv.Atoi(req.CmdArgs[1])
		set      search.Set
	)

	if state < proto.GroupUnknown || state > proto.GroupInitialized {
		return ErrInvalidSetState
	}

	if set, err = s.Sets.Get(ctx, name); err != nil {
		return
	}

	err = set.SetState(ctx, search.SetState(state))
	return
}

//---------------------------------------------------------------------------------//

type PostSets_AddReq struct {
	CmdArgs  []string
	Features []proto.FeatureJson `json:"features"`
}

func (s *Service) PostSets_Add(ctx context.Context, req *PostSets_AddReq, env *authstub.Env) (err error) {
	var (
		name = search.SetName(req.CmdArgs[0])
		set  search.Set
	)
	if set, err = s.Sets.Get(ctx, name); err != nil {
		return
	}

	_, xl := ctxAndLog(ctx, env.W, env.Req)
	var features []proto.Feature
	for _, fj := range req.Features {
		features = append(features, fj.ToFeature())
	}

	if err = set.Add(ctx, features...); err != nil {
		return httputil.NewError(http.StatusInternalServerError, err.Error())
	}

	s.Mutex.Lock()
	defer s.Mutex.Unlock()
	if recycle, exist := s.Recycles[name]; exist && recycle.Lifetime > 0 {
		for _, feature := range features {
			recycle.Timestamps = append(recycle.Timestamps, struct {
				ID        proto.FeatureID
				Timestamp time.Time
			}{
				ID:        feature.ID,
				Timestamp: time.Now(),
			})
		}
		s.Recycles[name] = recycle
	}

	var ids []proto.FeatureID
	for _, feature := range req.Features {
		ids = append(ids, feature.ID)
	}

	xl.Debugf("Add %#v features into set [%s]", ids, name)
	return
}

type PostSets_DeleteReq struct {
	CmdArgs []string
	IDs     []proto.FeatureID `json:"ids"`
}

type PostSets_DeleteResp struct {
	Deleted []proto.FeatureID `json:"deleted"`
}

func (s *Service) PostSets_Delete(ctx context.Context, req *PostSets_DeleteReq, env *authstub.Env) (resp PostSets_DeleteResp, err error) {
	var (
		name = search.SetName(req.CmdArgs[0])
		set  search.Set
	)
	if len(req.IDs) == 0 {
		err = ErrInvalidFeautres
		return
	}
	if set, err = s.Sets.Get(ctx, name); err != nil {
		return
	}

	_, xl := ctxAndLog(ctx, env.W, env.Req)

	if resp.Deleted, err = set.Delete(ctx, req.IDs...); err != nil {
		err = httputil.NewError(http.StatusInternalServerError, err.Error())
	}
	xl.Debugf("delete %v features from set [%s]", resp.Deleted, name)
	return
}

type PostSets_UpdateReq struct {
	CmdArgs  []string
	Features []proto.Feature `json:"features"`
}

type PostSets_UpdateResp struct {
	updated []proto.FeatureID `json:"updated"`
}

func (s *Service) PostSets_Update(ctx context.Context, req *PostSets_UpdateReq, env *authstub.Env) (resp PostSets_UpdateResp, err error) {
	// TODO

	return
}

type PostSets_DestroyReq struct {
	CmdArgs []string
}

func (s *Service) PostSets_Destroy(ctx context.Context, req *PostSets_DestroyReq, env *authstub.Env) (err error) {

	var (
		name = search.SetName(req.CmdArgs[0])
		set  search.Set
	)

	if set, err = s.Sets.Get(ctx, name); err != nil {
		return
	}

	_, xl := ctxAndLog(ctx, env.W, env.Req)
	if err = set.Destroy(ctx); err != nil {
		err = httputil.NewError(http.StatusInternalServerError, err.Error())
	}

	s.Mutex.Lock()
	defer s.Mutex.Unlock()
	recycle, exist := s.Recycles[name]
	if exist {
		recycle.Lifetime = 0
	}
	xl.Debugf("PostSets_Destroy: destory set %s", name)
	return
}

type PostSets_SearchReq struct {
	CmdArgs   []string
	Features  []proto.FeatureValue `json:"features"`
	Threshold float32              `json:"threshold"`
	Limit     int                  `json:"limit"`
}

type PostSets_SearchResp struct {
	SearchResults [][]feature_group.FeatureSearchItem `json:"search_results"`
}

func (s *Service) PostSets_Search(ctx context.Context, req *PostSets_SearchReq, env *authstub.Env) (resp PostSets_SearchResp, err error) {

	var (
		name = search.SetName(req.CmdArgs[0])
		set  search.Set
		xl   = xlog.FromContextSafe(ctx)
	)
	resp.SearchResults = make([][]feature_group.FeatureSearchItem, 0)

	if len(req.Features) == 0 {
		err = ErrInvalidFeautres
		return
	}

	if req.Limit == 0 {
		req.Limit = maxLimit
	}
	if set, err = s.Sets.Get(ctx, name); err != nil {
		return
	}

	rets, err := set.Search(ctx, req.Threshold, req.Limit, req.Features...)
	if err != nil {
		return
	}

	for _, ret := range rets {
		resp.SearchResults = append(resp.SearchResults, ret)
	}

	xl.Debugf("PostSets_Search search result: %#v", resp.SearchResults)

	return
}

type GetSets_Features_Req struct {
	CmdArgs []string
}

type GetSets_Features_Resp struct {
	Value proto.FeatureValueJson `json:"value,omitempty"`
}

func (s *Service) GetSets_Features_(ctx context.Context, req *GetSets_Features_Req, env *authstub.Env) (resp GetSets_Features_Resp, err error) {
	var (
		name = search.SetName(req.CmdArgs[0])
		id   = proto.FeatureID(req.CmdArgs[1])
		set  search.Set
	)

	if set, err = s.Sets.Get(ctx, name); err != nil {
		return
	}

	value, err := set.Get(ctx, id)
	if err != nil {
		err = httputil.NewError(http.StatusBadRequest, err.Error())
	}

	resp.Value = value.ToFeatureValueJson()
	return
}
