package eval

import (
	"context"
	"fmt"
	"io"
	"io/ioutil"
	"net/http"
	"os"
	"strings"
	"sync"
	"time"

	"github.com/qiniu/http/httputil.v1"
	"github.com/qiniu/http/restrpc.v1"
	"github.com/qiniu/xlog.v1"

	"qiniu.com/argus/atserving/model"
	"qiniu.com/argus/com/uri"
)

type Service interface {
	PostEval(context.Context, *postEvalArgs, *restrpc.Env)
	PostBatch(context.Context, *postEvalArgs, *restrpc.Env)
}

// service ...
type service struct {
	handler     Handler
	*sync.Mutex // 推理串行化
}

// NewService ...
func NewService(handler Handler, mustSerial bool) Service {
	s := &service{handler: handler}
	if mustSerial {
		s.Mutex = new(sync.Mutex)
	}
	return s
}

type postEvalArgs struct {
	ReqBody io.ReadCloser
}

func (s *service) initContext(ctx context.Context, env *restrpc.Env) (*xlog.Logger, context.Context) {
	xl, ok := xlog.FromContext(ctx)
	if !ok {
		xl = xlog.New(env.W, env.Req)
		ctx = xlog.NewContext(ctx, xl)
	}
	return xl, ctx
}

func (s *service) eval(
	ctx context.Context,
	reqs0 []model.EvalRequest,
	start time.Time,
	durations map[string]time.Duration,
) ([]EvalResponseInner, error) {

	var xl = xlog.FromContextSafe(ctx)
	var streams = make([]Stream, 0, len(reqs0))
	var images = make([]interface{}, 0)
	defer func() {
		for _, img := range images {
			switch v := img.(type) {
			case string:
				os.Remove(strings.TrimPrefix(v, _SchemeFilePrefix))
			}
		}
	}()

	var reqs []model.EvalRequestInner = make([]model.EvalRequestInner, 0, len(reqs0))

	for _, req := range reqs0 {
		streams = append(streams, newServiceStream(req.Data.URI.String()))
		reqs = append(reqs, model.ToEvalRequestInner(req))
	}
	images, err := s.handler.LoadEval(ctx, streams)
	if err != nil {
		xl.Error("[PostEval] LoadEval failed, err: ", err.Error())
		return nil, err
	}

	for i, image := range images {
		switch v := image.(type) {
		case string:
			reqs[i].Data.URI = model.STRING(v)
		case model.BYTES:
			reqs[i].Data.URI = v
		}
	}
	for i, _req := range reqs {
		_req, err := s.handler.PreEval(ctx, _req)
		if err != nil {
			xl.Error("[PostEval] PreEval failed, err: ", err.Error())
			return nil, err
		}
		reqs[i] = _req
		defer func() {
			switch v := _req.Data.URI.(type) {
			case model.STRING:
				os.Remove(strings.TrimPrefix(v.String(), _SchemeFilePrefix))
			}
		}()
	}
	durations["pre_eval"] = time.Since(start)
	start = time.Now()
	if s.Mutex != nil {
		s.Lock()
		defer s.Unlock()
	}
	resps, err := s.handler.Eval(ctx, reqs)
	if err != nil {
		xl.Errorf("[PostEval] Eval failed, err: %v", err)
		return nil, err
	}
	durations["eval"] = time.Since(start)
	xl.Infof("[EVAL] durations: [%v]", durations)
	return resps, nil
}

func (s *service) groupEval(
	ctx context.Context,
	reqs0 []model.GroupEvalRequest,
	start time.Time,
	durations map[string]time.Duration,
) ([]EvalResponseInner, error) {

	var xl = xlog.FromContextSafe(ctx)
	var streams = make([][]Stream, 0, len(reqs0))
	var images = make([][]interface{}, 0)
	defer func() {
		for _, bimgs := range images {
			for _, img := range bimgs {
				switch v := img.(type) {
				case string:
					os.Remove(strings.TrimPrefix(v, _SchemeFilePrefix))
				}
			}
		}
	}()

	var reqs = make([]model.GroupEvalRequestInner, 0, len(reqs0))

	for _, req := range reqs0 {
		sa := make([]Stream, 0, len(req.Data))
		for _, data := range req.Data {
			sa = append(sa, newServiceStream(data.URI.String()))
		}
		streams = append(streams, sa)
		reqs = append(reqs, model.ToGroupEvalRequestInner(req))
	}
	images, err := s.handler.LoadGroupEval(ctx, streams)
	if err != nil {
		xl.Error("[PostEval] LoadGroupEval failed, err: ", err.Error())
		return nil, err
	}
	for i, image := range images {
		for j := range reqs0[i].Data {
			switch v := image[j].(type) {
			case model.BYTES:
				reqs[i].Data[j].URI = v
			case string:
				reqs[i].Data[j].URI = model.STRING(v)
			}
		}
	}
	for i, _req := range reqs {
		_req, err := s.handler.PreGroupEval(ctx, _req)
		if err != nil {
			xl.Error("[PostEval] PreEval failed, err: ", err.Error())
			return nil, err
		}
		reqs[i] = _req
		defer func(_req model.GroupEvalRequestInner) {
			for _, data := range _req.Data {
				switch v := data.URI.(type) {
				case model.STRING:
					os.Remove(strings.TrimPrefix(v.String(), _SchemeFilePrefix))
				}
			}
		}(_req)
	}

	durations["pre_groupeval"] = time.Since(start)
	start = time.Now()
	if s.Mutex != nil {
		s.Lock()
		defer s.Unlock()
	}
	resps, err := s.handler.GroupEval(ctx, reqs)
	if err != nil {
		xl.Error("[PostEval] GroupEval failed, err: ", err.Error())
		return nil, err
	}
	durations["groupeval"] = time.Since(start)
	xl.Infof("[EVAL] durations: [%v]", durations)
	return resps, nil
}

// PostEval ...
func (s *service) PostEval(ctx context.Context, args *postEvalArgs, env *restrpc.Env) {
	xl, ctx := s.initContext(ctx, env)
	durations := make(map[string]time.Duration, 0)

	start := time.Now()
	var (
		eReq model.TaskReq
	)

	switch ct := env.Req.Header.Get(model.CONTENT_TYPE); {
	case model.IsJsonContent(ct):
		bs, err := ioutil.ReadAll(args.ReqBody)
		if err != nil {
			xl.Warnf("read requests body failed. %v", err)
			httputil.ReplyErr(env.W, http.StatusBadRequest, err.Error())
			return
		}
		if eReq, err = model.UnmarshalTaskRequest(ctx, bs); err != nil {
			xl.Warnf("decode request body failed. %s", err)
			httputil.ReplyErr(env.W, http.StatusBadRequest, err.Error())
			return
		}
		switch eReq.(type) {
		case model.EvalRequest:
			resps, err := s.eval(ctx,
				[]model.EvalRequest{eReq.(model.EvalRequest)},
				start, durations)
			if err != nil {
				code, desc := httputil.DetectError(err)
				httputil.ReplyErr(env.W, code, desc)
				return
			}
			if len(resps) > 0 && resps[0].Stream != nil {
				defer resps[0].Stream.Clean()
				st, length, _ := resps[0].Stream.Open(ctx)
				defer st.Close()
				env.W.Header().Set("Content-Length", fmt.Sprintf("%d", length))
				env.W.Header().Set("Content-Type", "application/octet-stream")
				io.Copy(env.W, st)
				return
			}
			if resps[0].Code >= 300 {
				xl.Warnf("eval failed, request:%#v, response %#v", eReq, resps[0])
				httputil.ReplyErr(env.W, resps[0].Code, resps[0].Message)
				return
			}
			httputil.Reply(env.W, http.StatusOK, resps[0])
			return
		case model.GroupEvalRequest:
			resps, err := s.groupEval(ctx,
				[]model.GroupEvalRequest{eReq.(model.GroupEvalRequest)}, start, durations)
			if err != nil {
				code, desc := httputil.DetectError(err)
				httputil.ReplyErr(env.W, code, desc)
				return
			}
			httputil.Reply(env.W, http.StatusOK, resps[0])
			return
		}
	case ct == model.CT_STREAM: // TODO
	default:
		xl.Warnf("bad content type. %s", ct)
		httputil.ReplyErr(env.W, http.StatusBadRequest, "wrong content type")
		return
	}
	return
}

// PostBatch ...
func (s *service) PostBatch(ctx context.Context, args *postEvalArgs, env *restrpc.Env) {
	xl, ctx := s.initContext(ctx, env)
	durations := make(map[string]time.Duration, 0)

	start := time.Now()
	var (
		eReqs []model.TaskReq
		ct    = env.Req.Header.Get(model.CONTENT_TYPE)
	)

	if !model.IsJsonContent(ct) {
		xl.Warnf("bad content type. %s", ct)
		httputil.ReplyErr(env.W, http.StatusBadRequest, "wrong content type")
		return
	}
	bs, err := ioutil.ReadAll(args.ReqBody)
	if err != nil {
		xl.Warnf("read requests body failed. %v", err)
		httputil.ReplyErr(env.W, http.StatusBadRequest, err.Error())
		return
	}
	if eReqs, err = model.UnmarshalBatchTaskRequest(ctx, bs); err != nil {
		xl.Warnf("decode request body failed. %s", err)
		httputil.ReplyErr(env.W, http.StatusBadRequest, err.Error())
		return
	}

	var (
		reqs1 = make([]model.EvalRequest, 0, len(eReqs))
		reqs2 = make([]model.GroupEvalRequest, 0, len(eReqs))
	)
	for _, eReq := range eReqs {
		switch eReq.(type) {
		case model.EvalRequest:
			if len(reqs2) > 0 {
				xl.Warnf("parse request body failed. %s", err)
				httputil.ReplyErr(env.W, http.StatusBadRequest, err.Error())
				return
			}
			reqs1 = append(reqs1, eReq.(model.EvalRequest))
		case model.GroupEvalRequest:
			if len(reqs1) > 0 {
				xl.Warnf("parse request body failed. %s", err)
				httputil.ReplyErr(env.W, http.StatusBadRequest, err.Error())
				return
			}
			reqs2 = append(reqs2, eReq.(model.GroupEvalRequest))
		}
	}
	if len(reqs1) > 0 {
		resps, err := s.eval(ctx, reqs1, start, durations)
		if err != nil {
			code, desc := httputil.DetectError(err)
			httputil.ReplyErr(env.W, code, desc)
			return
		}
		httputil.Reply(env.W, http.StatusOK, resps)
		return
	}
	if len(reqs2) > 0 {
		resps, err := s.groupEval(ctx, reqs2, start, durations)
		if err != nil {
			code, desc := httputil.DetectError(err)
			httputil.ReplyErr(env.W, code, desc)
			return
		}
		httputil.Reply(env.W, http.StatusOK, resps)
		return
	}
	return
}

//----------------------------------------------------------------------------//

type serviceStream struct {
	uri string
}

func newServiceStream(uri string) Stream { return serviceStream{uri: uri} }

func (h serviceStream) Name() string { return h.uri }

func (h serviceStream) Open(ctx context.Context) (io.ReadCloser, int64, error) {
	c := uri.New(
		uri.WithHTTPHandler(),
		uri.WithFileHandler(), // 支持本地文件读取
		uri.WithDataHandler(), // 支持data:application/octet-stream;base64,
	) // TODO Add uri.WithAdminAkSk()
	resp, err := c.Get(ctx, uri.Request{URI: h.uri})
	if err != nil {
		return nil, 0, err
	}
	return resp.Body, resp.Size, nil
}

func (h serviceStream) Clean() error { return nil }
