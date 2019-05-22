package gate

import (
	"context"
	"io"
	"io/ioutil"
	"net/http"

	"github.com/qiniu/http/httputil.v1"
	"github.com/qiniu/xlog.v1"

	"qiniu.com/argus/atserving/model"
	"qiniu.com/argus/com/util"
	"qiniu.com/auth/authstub.v1"
)

// BaseEvalReq ...
type BaseEvalReq struct {
	CmdArgs []string
	ReqBody io.ReadCloser

	Timeout *string `json:"timeout"` // TODO
}

// Server ...
type Server interface {
	// POST /eval/<cmd>
	//
	// Response Json OR Stream
	PostEval_(context.Context, *BaseEvalReq, *authstub.Env)
	// POST /eval/<cmd>/<version>
	//
	// Response Json OR Stream
	PostEval__(context.Context, *BaseEvalReq, *authstub.Env)
	// POST /batch
	PostBatch(context.Context, *BaseEvalReq, *authstub.Env) ([]interface{}, error)
}

var _ Server = &server{}

type server struct {
	Gate
	Evals
	logPush *LogPushClient
}

// NewServer ...
func NewServer(gate Gate, evals Evals, logPush *LogPushClient) Server {
	return &server{
		Gate:    gate,
		Evals:   evals,
		logPush: logPush,
	}
}

func (s *server) initContext(ctx context.Context, env *authstub.Env) (*xlog.Logger, context.Context) {
	xl, ok := xlog.FromContext(ctx)
	if !ok {
		xl = xlog.New(env.W, env.Req)
		ctx = xlog.NewContext(ctx, xl)
	}
	return xl, ctx
}

func (s *server) PostEval_(ctx context.Context, req *BaseEvalReq, env *authstub.Env) {
	s.postEval(ctx, req, env)
}
func (s *server) PostEval__(ctx context.Context, req *BaseEvalReq, env *authstub.Env) {
	s.postEval(ctx, req, env)
}
func (s *server) postEval(ctx context.Context, req *BaseEvalReq, env *authstub.Env) {
	xl, ctx := s.initContext(ctx, env)

	var (
		cmd     = req.CmdArgs[0]
		version *string
		eReq    model.TaskReq
		eResp   model.EvalResponse
		rc      io.ReadCloser
		length  int64
		header  http.Header
		err     error
	)
	if len(req.CmdArgs) >= 2 {
		version = &req.CmdArgs[1]
	}

	al := newAiprdlog("SERVING-GATE-EVAL")
	defer func() {
		// xl.Info(string(al.marshal()))
		go s.logPush.sendLog(util.SpawnContextOnlyReqID(ctx), al)
	}()

	al.addBase(xl.ReqId(), env.Uid, cmd, version)

	xl.Infof("post eval. %s %s", cmd, p2s(version))

	if !s.IsAllowable(env.Uid, cmd) {
		xl.Warnf("not accept. %d %s", env.Uid, cmd)
		httputil.ReplyErr(env.W, ErrNotAcceptable.Code, ErrNotAcceptable.Err)
		return
	}

	switch ct := env.Req.Header.Get(model.CONTENT_TYPE); {
	case model.IsJsonContent(ct):
		var bs []byte
		bs, err = ioutil.ReadAll(req.ReqBody)
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
			v := eReq.(model.EvalRequest)
			_uri, _ := improveURI(v.Data.URI.String(), env.Uid)
			v.Data.URI = model.STRING(_uri)
			al.addUri(_uri)
			v.Cmd = cmd
			v.Version = version
			eReq = v
			al.addEvalRequest(&v)
		case model.GroupEvalRequest:
			v := eReq.(model.GroupEvalRequest)
			for i := range v.Data {
				_uri, _ := improveURI(v.Data[i].URI.String(), env.Uid)
				v.Data[i].URI = model.STRING(_uri)
				al.addUri(_uri)
			}
			v.Cmd = cmd
			v.Version = version
			eReq = v
			al.addGroupEvalRequest(&v)
		}
		xl.Infof("post eval. %s %#v", cmd, eReq)
		eResp, rc, length, header, err = s.Eval(ctx, eReq)
		al.addEvalResponse(&eResp)
	case ct == model.CT_STREAM:
		eResp, rc, length, header, err = s.EvalBody(ctx, cmd, version, env.Req.Body, env.Req.ContentLength)
	default:
		xl.Warnf("bad content type. %s", ct)
		httputil.ReplyErr(env.W, http.StatusBadRequest, "wrong content type")
		return
	}

	if err != nil {
		xl.Errorf("eval failed. %s", err)
		code, desc := httputil.DetectError(err)
		httputil.ReplyErr(env.W, code, desc)
		return
	}

	mergeHeader(env.Utype, env.W.Header(), header)

	if rc != nil {
		defer rc.Close()
		xl.Infof("eval response stream. %d", length)
		httputil.ReplyWithStream(env.W, http.StatusOK, model.CT_STREAM, rc, length)
		return
	}

	xl.Infof("eval response json. %#v", eResp)
	httputil.Reply(env.W, http.StatusOK, eResp)
}

func (s *server) PostBatch(
	ctx context.Context,
	req *BaseEvalReq,
	env *authstub.Env,
) (resp []interface{}, err error) {

	xl, ctx := s.initContext(ctx, env)

	var (
		reqs   []model.TaskReq
		header = make(http.Header)
		als    []*aiprdLog
	)

	xl.Info("post eval batch.")

	switch ct := env.Req.Header.Get(model.CONTENT_TYPE); {
	case model.IsJsonContent(ct):
		var bs []byte
		bs, err = ioutil.ReadAll(req.ReqBody)
		if err != nil {
			xl.Warnf("read requests body failed. %v", err)
			return nil, httputil.NewError(http.StatusBadRequest, err.Error())
		}
		if reqs, err = model.UnmarshalBatchTaskRequest(ctx, bs); err != nil {
			xl.Warnf("decode request body failed. %s", err)
			return nil, httputil.NewError(http.StatusBadRequest, err.Error())
		}
		for range reqs {
			als = append(als, newAiprdlog("SERVING-GATE-EVALBATCH"))
		}

		for i, req := range reqs {
			switch req.(type) {
			case model.EvalRequest:
				v := req.(model.EvalRequest)
				_uri, _ := improveURI(v.Data.URI.String(), env.Uid)
				als[i].addUri(_uri)
				v.Data.URI = model.STRING(_uri)
				v.Cmd, v.Version, err = parseOP(v.OP)
				if err != nil {
					xl.Warnf("parse op failed. %s %v", v.OP, err)
					return nil, err
				}
				reqs[i] = v
				als[i].addEvalRequest(&v)
				als[i].addBase(xl.ReqId(), env.Uid, v.Cmd, v.Version)
			case model.GroupEvalRequest:
				v := req.(model.GroupEvalRequest)
				for j := range v.Data {
					_uri, _ := improveURI(v.Data[j].URI.String(), env.Uid)
					als[i].addUri(_uri)
					v.Data[j].URI = model.STRING(_uri)
				}
				v.Cmd, v.Version, err = parseOP(v.OP)
				if err != nil {
					xl.Warnf("parse op failed. %s %v", v.OP, err)
					return nil, err
				}
				reqs[i] = v
				als[i].addGroupEvalRequest(&v)
			}
		}

		for _, req := range reqs {
			if !s.IsAllowable(env.Uid, req.GetCmd()) {
				xl.Warnf("not accept. %d %s", env.Uid, req.GetCmd())
				err = httputil.NewError(http.StatusNotAcceptable, req.GetCmd())
				return
			}
		}

		resp, header, err = s.EvalBatch(ctx, reqs)

		for i, al := range als {
			if i < len(resp) {
				al.addEvalBatchResponse(resp[i])
				// xl.Info(string(al.marshal()))
				go s.logPush.sendLog(util.SpawnContextOnlyReqID(ctx), al)
			}
		}
	default:
		xl.Warnf("wrong content type. %s", ct)
		err = httputil.NewError(http.StatusBadGateway, "wrong content type")
		return
	}

	mergeHeader(env.Utype, env.W.Header(), header)

	if err == nil {
		xl.Infof("eval batch. %#v", resp)
	} else {
		xl.Errorf("eval batch failed. %s", err)
	}
	return
}

func mergeHeader(utype uint32, h1, h2 http.Header) {
	measure := model.NewMeasure(model.XOriginA)
	model.NewHeaderMerger(
		model.NewHeaderValueFunc(func(key, v1, v2 string) (bool, string) {
			if utype == 0 {
				return false, ""
			}
			return measure.Merge(key, v1, v2)
		}),
	).Merge(h1, h2)
}
