package utility

import (
	"context"
	"strings"
	"time"

	"qiniu.com/auth/authstub.v1"
)

type OcrSariVatReq struct {
	Data struct {
		URI string `json:"uri"`
	} `json:"data"`
}

type OcrSariVatResp struct {
	Code    int              `json:"code"`
	Message string           `json:"message"`
	Result  OcrSariVatResult `json:"result"`
}

type OcrSariVatResult struct {
	URI    string                 `json:"uri"`
	Bboxes [][4][2]float32        `json:"bboxes"`
	Res    map[string]interface{} `json:"res"`
}

type _EvalOcrSariVatDetectReq struct {
	Data struct {
		URI string `json:"uri"`
	} `json:"data"`
	Params struct {
		Type string `json:"type"`
	} `json:"params"`
}

type _EvalOcrSariVatDetectResp struct {
	Code    int    `json:"code"`
	Message string `json:"message"`
	Result  struct {
		Bboxes [][4][2]float32 `json:"bboxes"`
	} `json:"result"`
}

type _EvalOcrSariVatRecogReq struct {
	Data struct {
		URI string `json:"uri"`
	} `json:"data"`
	Params struct {
		Bboxes [][4][2]float32 `json:"bboxes"`
	} `json:"params"`
}

type _EvalOcrSariVatRecogResp struct {
	Code    int    `json:"code"`
	Message string `json:"message"`
	Result  struct {
		Texts []string `json:"text"`
	} `json:"result"`
}

type _EvalOcrSariVatPostProcessReq struct {
	Data struct {
		URI string `json:"uri"`
	} `json:"data"`
	Params struct {
		Type  string   `json:"type"`
		Texts []string `json:"texts"`
	} `json:"params"`
}

type _EvalOcrSariVatPostProcessResp struct {
	Code    int                    `json:"code"`
	Message string                 `json:"message"`
	Result  map[string]interface{} `json:"result"`
}

type iOcrSariVat interface {
	SariVatPostProcessEval(context.Context, _EvalOcrSariVatPostProcessReq, _EvalEnv) (_EvalOcrSariVatPostProcessResp, error)
	SariVatDetectEval(context.Context, _EvalOcrSariVatDetectReq, _EvalEnv) (_EvalOcrSariVatDetectResp, error)
	SariVatRecognizeEval(context.Context, _EvalOcrSariVatRecogReq, _EvalEnv) (_EvalOcrSariVatRecogResp, error)
}

type _OcrSariVat struct {
	host    string
	timeout time.Duration
}

func newOcrSariVat(host string, timeout time.Duration) iOcrSariVat {
	return _OcrSariVat{host: host, timeout: timeout}
}

func (ot _OcrSariVat) SariVatPostProcessEval(
	ctx context.Context, req _EvalOcrSariVatPostProcessReq, env _EvalEnv,
) (_EvalOcrSariVatPostProcessResp, error) {
	var (
		url    = ot.host + "/v1/eval/ocr-sari-vat"
		client = newRPCClient(env, ot.timeout)
		resp   _EvalOcrSariVatPostProcessResp
	)
	err := callRetry(ctx,
		func(ctx context.Context) error {
			return client.CallWithJson(ctx, &resp, "POST", url, &req)
		})
	return resp, err
}

func (ot _OcrSariVat) SariVatDetectEval(
	ctx context.Context, req _EvalOcrSariVatDetectReq, env _EvalEnv,
) (_EvalOcrSariVatDetectResp, error) {
	var (
		url    = ot.host + "/v1/eval/ocr-sari-vat"
		client = newRPCClient(env, ot.timeout)
		resp   _EvalOcrSariVatDetectResp
	)
	err := callRetry(ctx,
		func(ctx context.Context) error {
			return client.CallWithJson(ctx, &resp, "POST", url, &req)
		})
	return resp, err
}

func (ot _OcrSariVat) SariVatRecognizeEval(
	ctx context.Context, req _EvalOcrSariVatRecogReq, env _EvalEnv,
) (_EvalOcrSariVatRecogResp, error) {
	var (
		url    = ot.host + "/v1/eval/ocr-sari-crann"
		client = newRPCClient(env, ot.timeout)
		resp   _EvalOcrSariVatRecogResp
	)
	err := callRetry(ctx,
		func(ctx context.Context) error {
			return client.CallWithJson(ctx, &resp, "POST", url, &req)
		})
	return resp, err
}

func (s *Service) PostOcrVat(ctx context.Context, args *OcrSariVatReq, env *authstub.Env) (ret *OcrSariVatResp, err error) {
	var (
		uid     = env.UserInfo.Uid
		utype   = env.UserInfo.Utype
		evalEnv = _EvalEnv{Uid: uid, Utype: utype}
	)
	ret = &OcrSariVatResp{}
	ctx, xl := ctxAndLog(ctx, env.W, env.Req)

	if strings.TrimSpace(args.Data.URI) == "" {
		xl.Error("empty data.uri")
		return nil, ErrArgs
	}

	// detect
	var (
		otDetectReq  _EvalOcrSariVatDetectReq
		otDetectResp _EvalOcrSariVatDetectResp
	)
	otDetectReq.Data.URI = args.Data.URI
	otDetectReq.Params.Type = "detect"
	otDetectResp, err = s.iOcrSariVat.SariVatDetectEval(ctx, otDetectReq, evalEnv)
	if err != nil {
		xl.Errorf("vat ticket detect failed. %v", err)
		return
	}
	if len(otDetectResp.Result.Bboxes) == 0 {
		xl.Debugf("vat ticket detect got no bbox, ignore this image")
		return
	}
	xl.Infof("vat ticket detect got bboxes: %v", otDetectResp.Result.Bboxes)

	// recognize
	var (
		otRecognizeReq  _EvalOcrSariVatRecogReq
		otRecognizeResp _EvalOcrSariVatRecogResp
	)
	otRecognizeReq.Data.URI = args.Data.URI
	otRecognizeReq.Params.Bboxes = otDetectResp.Result.Bboxes
	otRecognizeResp, err = s.iOcrSariVat.SariVatRecognizeEval(ctx, otRecognizeReq, evalEnv)
	if err != nil {
		xl.Errorf("vat ticket recognize failed. %v", err)
		return
	}
	xl.Infof("vat ticket recognize got texts: %v", otRecognizeResp.Result.Texts)

	// post recognize
	var (
		otPostRecogReq  _EvalOcrSariVatPostProcessReq
		otPostRecogResp _EvalOcrSariVatPostProcessResp
	)
	otPostRecogReq.Data.URI = args.Data.URI
	otPostRecogReq.Params.Type = "postrecog"
	otPostRecogReq.Params.Texts = otRecognizeResp.Result.Texts
	otPostRecogResp, err = s.iOcrSariVat.SariVatPostProcessEval(ctx, otPostRecogReq, evalEnv)
	if err != nil {
		xl.Errorf("vat ticket post-recognize failed. %v", err)
		return
	}

	xl.Infof("vat ticket post-recognize got result: %v", otPostRecogResp.Result)

	ret.Result.Bboxes = otDetectResp.Result.Bboxes
	ret.Result.Res = otPostRecogResp.Result

	if err == nil && env.UserInfo.Utype != NoChargeUtype {
		setStateHeader(env.W.Header(), "OCR_VAT", 1)
	}

	return
}
