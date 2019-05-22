package utility

import (
	"context"
	"strings"
	"time"

	"qiniu.com/auth/authstub.v1"
)

type OcrSariIdcardReq struct {
	Data struct {
		URI string `json:"uri"`
	} `json:"data"`
}

type OcrSariIdcardResp struct {
	Code    int                 `json:"code"`
	Message string              `json:"message"`
	Result  OcrSariIdcardResult `json:"result"`
}

type OcrSariIdcardResult struct {
	URI    string            `json:"uri"`
	Bboxes [][4][2]int       `json:"bboxes"`
	Type   int               `json:"type"`
	Res    map[string]string `json:"res"`
}

type _EvalOcrSariIdcardDetectReq struct {
	Data struct {
		URI string `json:"uri"`
	} `json:"data"`
}

type _EvalOcrSariIdcardDetectResp struct {
	Code    int    `json:"code"`
	Message string `json:"message"`
	Result  struct {
		Bboxes [][8]int `json:"bboxes"`
	} `json:"result"`
}

type _EvalOcrSariIdcardRecogReq struct {
	Data struct {
		URI string `json:"uri"`
	} `json:"data"`
	Params struct {
		Bboxes [][4][2]int `json:"bboxes"`
	} `json:"params"`
}

type _EvalOcrSariIdcardRecogResp struct {
	Code    int    `json:"code"`
	Message string `json:"message"`
	Result  struct {
		Texts []string `json:"text"`
	} `json:"result"`
}

type _EvalOcrSariIdcardPreProcessReq struct {
	Data struct {
		URI string `json:"uri"`
	} `json:"data"`
	Params struct {
		Type          string      `json:"type"`
		Bboxes        [][4][2]int `json:"bboxes"`
		Class         int         `json:"class"`
		Texts         []string    `json:"texts"`
		Names         []string    `json:"names"`
		Regions       [][4][2]int `json:"regions"`
		DetectedBoxes [][8]int    `json:"detectedBoxes"`
	} `json:"params"`
}

type _EvalOcrSariIdcardPreProcessResp struct {
	Code    int    `json:"code"`
	Message string `json:"message"`
	Result  struct {
		Class         int               `json:"class"`
		AlignedImg    string            `json:"alignedImg"`
		Names         []string          `json:"names"`
		Regions       [][4][2]int       `json:"regions"`
		Bboxes        [][4][2]int       `json:"bboxes"`
		DetectedBoxes [][8]int          `json:"detectedBoxes"`
		Res           map[string]string `json:"res"`
	} `json:"result"`
}

type iOcrSariIdcard interface {
	SariIdcardPreProcessEval(context.Context, _EvalOcrSariIdcardPreProcessReq, _EvalEnv) (_EvalOcrSariIdcardPreProcessResp, error)
	SariIdcardDetectEval(context.Context, _EvalOcrSariIdcardDetectReq, _EvalEnv) (_EvalOcrSariIdcardDetectResp, error)
	SariIdcardRecognizeEval(context.Context, _EvalOcrSariIdcardRecogReq, _EvalEnv) (_EvalOcrSariIdcardRecogResp, error)
}

type _OcrSariIdcard struct {
	host    string
	timeout time.Duration
}

func newOcrSariIdcard(host string, timeout time.Duration) iOcrSariIdcard {
	return _OcrSariIdcard{host: host, timeout: timeout}
}

func (ot _OcrSariIdcard) SariIdcardPreProcessEval(
	ctx context.Context, req _EvalOcrSariIdcardPreProcessReq, env _EvalEnv,
) (_EvalOcrSariIdcardPreProcessResp, error) {
	var (
		url    = ot.host + "/v1/eval/ocr-sari-id-pre"
		client = newRPCClient(env, ot.timeout)
		resp   _EvalOcrSariIdcardPreProcessResp
	)
	err := callRetry(ctx,
		func(ctx context.Context) error {
			return client.CallWithJson(ctx, &resp, "POST", url, &req)
		})
	return resp, err
}

func (ot _OcrSariIdcard) SariIdcardDetectEval(
	ctx context.Context, req _EvalOcrSariIdcardDetectReq, env _EvalEnv,
) (_EvalOcrSariIdcardDetectResp, error) {
	var (
		url    = ot.host + "/v1/eval/ocr-scene-detect"
		client = newRPCClient(env, ot.timeout)
		resp   _EvalOcrSariIdcardDetectResp
	)
	err := callRetry(ctx,
		func(ctx context.Context) error {
			return client.CallWithJson(ctx, &resp, "POST", url, &req)
		})
	return resp, err
}

func (ot _OcrSariIdcard) SariIdcardRecognizeEval(
	ctx context.Context, req _EvalOcrSariIdcardRecogReq, env _EvalEnv,
) (_EvalOcrSariIdcardRecogResp, error) {
	var (
		url    = ot.host + "/v1/eval/ocr-sari-crann"
		client = newRPCClient(env, ot.timeout)
		resp   _EvalOcrSariIdcardRecogResp
	)
	err := callRetry(ctx,
		func(ctx context.Context) error {
			return client.CallWithJson(ctx, &resp, "POST", url, &req)
		})
	return resp, err
}

func (s *Service) PostOcrIdcard(
	ctx context.Context, args *OcrSariIdcardReq, env *authstub.Env,
) (ret *OcrSariIdcardResp, err error) {
	var (
		uid     = env.UserInfo.Uid
		utype   = env.UserInfo.Utype
		evalEnv = _EvalEnv{Uid: uid, Utype: utype}
	)
	ret = &OcrSariIdcardResp{}
	// ret.Result.Bboxes = make([][4][2]int, 0)
	// ret.Result.Text = make([]string, 0)
	ctx, xl := ctxAndLog(ctx, env.W, env.Req)

	if strings.TrimSpace(args.Data.URI) == "" {
		xl.Error("empty data.uri")
		return nil, ErrArgs
	}

	// pre detect
	var (
		otPreDetectReq  _EvalOcrSariIdcardPreProcessReq
		otPreDetectResp _EvalOcrSariIdcardPreProcessResp
	)
	otPreDetectReq.Data.URI = args.Data.URI
	otPreDetectReq.Params.Type = "predetect"
	otPreDetectResp, err = s.iOcrSariIdcard.SariIdcardPreProcessEval(ctx, otPreDetectReq, evalEnv)
	if err != nil {
		xl.Errorf("id card pre-detect failed. %v", err)
		return
	}
	if len(otPreDetectResp.Result.Bboxes) == 0 {
		xl.Debugf("id card pre-detect got no bbox, ignore this image")
		return
	}
	xl.Infof("id card pre-detect got class: %v", otPreDetectResp.Result.Class)
	xl.Infof("id card pre-detect got names: %v", otPreDetectResp.Result.Names)
	xl.Infof("id card pre-detect got bboxes: %v", otPreDetectResp.Result.Bboxes)
	xl.Infof("id card pre-detect got bboxes: %v", otPreDetectResp.Result.Regions)

	// detect
	var (
		otDetectReq  _EvalOcrSariIdcardDetectReq
		otDetectResp _EvalOcrSariIdcardDetectResp
	)
	otDetectReq.Data.URI = "data:application/octet-stream;base64," + otPreDetectResp.Result.AlignedImg
	otDetectResp, err = s.iOcrSariIdcard.SariIdcardDetectEval(ctx, otDetectReq, evalEnv)
	if err != nil {
		xl.Errorf("id card detect failed. %v", err)
		return
	}
	if len(otDetectResp.Result.Bboxes) == 0 {
		xl.Debugf("id card detect got not bbox, ignore this image")
		return
	}
	xl.Infof("id card detect got bboxes: %v", otDetectResp.Result.Bboxes)

	// pre recognize
	var (
		otPreRecogReq  _EvalOcrSariIdcardPreProcessReq
		otPreRecogResp _EvalOcrSariIdcardPreProcessResp
	)
	otPreRecogReq.Data.URI = otDetectReq.Data.URI
	otPreRecogReq.Params.Type = "prerecog"
	otPreRecogReq.Params.Class = otPreDetectResp.Result.Class
	otPreRecogReq.Params.DetectedBoxes = otDetectResp.Result.Bboxes
	otPreRecogReq.Params.Names = otPreDetectResp.Result.Names
	otPreRecogReq.Params.Bboxes = otPreDetectResp.Result.Bboxes
	otPreRecogReq.Params.Regions = otPreDetectResp.Result.Regions
	otPreRecogResp, err = s.iOcrSariIdcard.SariIdcardPreProcessEval(ctx, otPreRecogReq, evalEnv)
	if err != nil {
		xl.Errorf("id card pre-recognize failed. %v", err)
		return
	}
	if len(otPreRecogResp.Result.Bboxes) == 0 {
		xl.Debugf("id card pre-recognize got no bbox, ignore this image")
		return
	}
	xl.Infof("id card pre-recognize got bboxes: %v", otPreRecogResp.Result.Bboxes)

	// recognize
	var (
		otRecognizeReq  _EvalOcrSariIdcardRecogReq
		otRecognizeResp _EvalOcrSariIdcardRecogResp
	)
	otRecognizeReq.Data.URI = otDetectReq.Data.URI
	otRecognizeReq.Params.Bboxes = otPreRecogResp.Result.Bboxes
	otRecognizeResp, err = s.iOcrSariIdcard.SariIdcardRecognizeEval(ctx, otRecognizeReq, evalEnv)
	if err != nil {
		xl.Errorf("id card recognize failed. %v", err)
		return
	}
	xl.Infof("id card recognize got texts: %v", otRecognizeResp.Result.Texts)

	// post recognize
	var (
		otPostRecogReq  _EvalOcrSariIdcardPreProcessReq
		otPostRecogResp _EvalOcrSariIdcardPreProcessResp
	)
	otPostRecogReq.Data.URI = otDetectReq.Data.URI
	otPostRecogReq.Params.Type = "postprocess"
	otPostRecogReq.Params.Class = otPreDetectResp.Result.Class
	otPostRecogReq.Params.Bboxes = otPreRecogResp.Result.Bboxes
	otPostRecogReq.Params.Regions = otPreDetectResp.Result.Regions
	otPostRecogReq.Params.Names = otPreDetectResp.Result.Names
	otPostRecogReq.Params.Texts = otRecognizeResp.Result.Texts
	otPostRecogResp, err = s.iOcrSariIdcard.SariIdcardPreProcessEval(ctx, otPostRecogReq, evalEnv)
	if err != nil {
		xl.Errorf("id card post-recognize failed. %v", err)
		return
	}

	xl.Infof("id card post-recognize got result: %v", otPostRecogResp.Result.Res)

	ret.Result.URI = otPreDetectResp.Result.AlignedImg
	ret.Result.Bboxes = otPreRecogResp.Result.Bboxes
	ret.Result.Type = otPreDetectResp.Result.Class
	ret.Result.Res = otPostRecogResp.Result.Res

	if err == nil && env.UserInfo.Utype != NoChargeUtype {
		setStateHeader(env.W.Header(), "OCR_IDCARD", 1)
	}

	return
}
