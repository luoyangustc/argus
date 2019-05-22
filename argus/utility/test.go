package utility

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"time"

	"github.com/qiniu/http/httputil.v1"
	authstub "qiniu.com/auth/authstub.v1"

	ahttp "qiniu.com/argus/argus/com/http"
	"qiniu.com/argus/atserving/model"
)

type TestReq struct {
	ReqBody io.ReadCloser
}

func (s *Service) PostTestA(
	ctx context.Context,
	req *TestReq,
	env *authstub.Env,
) (ret []string, err error) {

	ctx, xl := ctxAndLog(ctx, env.W, env.Req)
	var (
		uid   = env.Uid
		utype = env.Utype
		sReq  = make([]struct {
			Data struct {
				URI string `json:"uri"`
			} `json:"data"`
		}, 0)

		decoder = json.NewDecoder(req.ReqBody)
	)

	if err = decoder.Decode(&sReq); err != nil {
		xl.Warnf("parse request body failed. %v", err)
		err = httputil.NewError(http.StatusBadRequest, err.Error())
		return
	}

	var (
		dReq   = make([]model.EvalRequest, 0, len(sReq))
		client = ahttp.NewQiniuStubRPCClient(uid, utype, time.Second*5)
	)
	for _, src := range sReq {
		dReq = append(dReq,
			model.EvalRequest{
				OP:   "/v1/eval/hello-eval",
				Data: model.Resource{URI: model.STRING(src.Data.URI)},
			},
		)
	}
	err = client.CallWithJson(ctx, &ret, "POST", s.ServingHost+"/v1/batch", dReq)
	if err != nil {
		xl.Errorf("call serving failed. %s %#v %v", s.ServingHost, dReq, err)
		err = httputil.NewError(http.StatusInternalServerError, err.Error())
		return
	}

	xl.Info("done.")
	return
}

type TestBResp struct {
	Code    int    `json:"code"`
	Message string `json:"message"`
	Result  int64  `json:"result"`
}

func (s *Service) PostTestB(
	ctx context.Context,
	req *TestReq,
	env *authstub.Env,
) (ret []TestBResp, err error) {

	ctx, xl := ctxAndLog(ctx, env.W, env.Req)
	var (
		uid   = env.Uid
		utype = env.Utype
		sReq  = make([]struct {
			Data struct {
				URI string `json:"uri"`
			} `json:"data"`
		}, 0)

		decoder = json.NewDecoder(req.ReqBody)
	)

	if err = decoder.Decode(&sReq); err != nil {
		xl.Warnf("parse request body failed. %v", err)
		err = httputil.NewError(http.StatusBadRequest, err.Error())
		return
	}

	var (
		dReq   = make([]model.GroupEvalRequest, 0, len(sReq))
		client = ahttp.NewQiniuStubRPCClient(uid, utype, time.Second*5)
		dResp  = make([]struct {
			Code    int     `json:"code"`
			Message string  `json:"message"`
			Result  []int64 `json:"result"`
		}, 0)
	)
	for _, src := range sReq {
		dReq = append(dReq,
			model.GroupEvalRequest{
				OP:   "/v1/eval/hello-eval",
				Data: []model.Resource{model.Resource{URI: model.STRING(src.Data.URI)}},
			},
		)
	}
	err = client.CallWithJson(ctx, &dResp, "POST", s.ServingHost+"/v1/batch", dReq)
	if err != nil {
		xl.Errorf("call serving failed. %s %#v %v", s.ServingHost, dReq, err)
		err = httputil.NewError(http.StatusInternalServerError, err.Error())
		return
	}
	for _, resp := range dResp {
		resp2 := TestBResp{
			Code:    resp.Code,
			Message: resp.Message,
		}
		if resp.Result != nil && len(resp.Result) > 0 {
			resp2.Result = resp.Result[0]
		}
		ret = append(ret, resp2)
	}

	xl.Info("done.")
	return
}
