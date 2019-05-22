package utility

import (
	"context"
	"strings"
	"time"

	authstub "qiniu.com/auth/authstub.v1"
)

const (
	IgnoreImageType string = "others"
	OtherTextScene  string = "other-text"
)

type OcrTextReq struct {
	Data struct {
		URI string `json:"uri"`
	} `json:"data"`
}

type OcrTextResp struct {
	Code    int           `json:"code"`
	Message string        `json:"message"`
	Result  OcrTextResult `json:"result"`
}

type OcrTextResult struct {
	Type   string      `json:"type"`
	Bboxes [][4][2]int `json:"bboxes"`
	Texts  []string    `json:"texts"`
}

type _EvalOcrTextClassifyReq OcrTextReq

type _EvalOcrTextClassifyResp struct {
	Code    int    `json:"code"`
	Message string `json:"message"`
	Result  struct {
		Confidences []struct {
			Index int     `json:"index"`
			Class string  `json:"class"`
			Score float64 `json:"score"`
		} `json:"confidences"`
	} `json:"result"`
}

type _EvalOcrCtpnReq struct {
	Data struct {
		URI string `json:"uri"`
	} `json:"data"`
}

type _EvalOcrCtpnResp struct {
	Code    int    `json:"code"`
	Message string `json:"message"`
	Result  struct {
		Bboxes [][4][2]int `json:"bboxes"`
	} `json:"result"`
}

type _EvalOcrTextRecognizeReq struct {
	Data struct {
		URI string `json:"uri"`
	} `json:"data"`
	Params struct {
		Bboxes    [][4]int `json:"bboxes"`
		ImageType string   `json:"image_type"`
	} `json:"params"`
}

type _EvalOcrTextRecognizeResp struct {
	Code    int    `json:"code"`
	Message string `json:"message"`
	Result  struct {
		Bboxes [][4][2]int `json:"bboxes"`
		Texts  []string    `json:"texts"`
	} `json:"result"`
}

type _EvalOcrSceneDetectReq struct {
	Data struct {
		URI string `json:"uri"`
	} `json:"data"`
}

type _EvalOcrSceneDetectResp struct {
	Code    int    `json:"code"`
	Message string `json:"message"`
	Result  struct {
		Bboxes [][8]int `json:"bboxes"`
	} `json:"result"`
}

type _EvalOcrSceneRecognizeReq struct {
	Data struct {
		URI string `json:"uri"`
	} `json:"data"`
	Params struct {
		Bboxes [][8]int `json:"bboxes"`
	} `json:"params"`
}

type _EvalOcrSceneRecognizeResp struct {
	Code    int    `json:"code"`
	Message string `json:"message"`
	Result  struct {
		Texts []OcrSceneRespResult `json:"texts"`
	} `json:"result"`
}

type OcrSceneRespResult struct {
	Bboxes [8]int `json:"bboxes"`
	Text   string `json:"text"`
}

type iOcrText interface {
	ClassifyEval(context.Context, _EvalOcrTextClassifyReq, _EvalEnv) (_EvalOcrTextClassifyResp, error)
	DetectEval(context.Context, _EvalOcrCtpnReq, _EvalEnv) (_EvalOcrCtpnResp, error)
	RecognizeEval(context.Context, _EvalOcrTextRecognizeReq, _EvalEnv) (_EvalOcrTextRecognizeResp, error)
	SceneDetectEval(context.Context, _EvalOcrSceneDetectReq, _EvalEnv) (_EvalOcrSceneDetectResp, error)
	SceneRecognizeEval(context.Context, _EvalOcrSceneRecognizeReq, _EvalEnv) (_EvalOcrSceneRecognizeResp, error)
}

type _OcrText struct {
	host    string
	timeout time.Duration
}

func newOcrText(host string, timeout time.Duration) iOcrText {
	return _OcrText{host: host, timeout: timeout}
}

func (ot _OcrText) ClassifyEval(
	ctx context.Context, req _EvalOcrTextClassifyReq, env _EvalEnv,
) (_EvalOcrTextClassifyResp, error) {
	var (
		url    = ot.host + "/v1/eval/ocr-classify"
		client = newRPCClient(env, ot.timeout)
		resp   _EvalOcrTextClassifyResp
	)
	err := callRetry(ctx,
		func(ctx context.Context) error {
			return client.CallWithJson(ctx, &resp, "POST", url, &req)
		})
	return resp, err
}

func (ot _OcrText) DetectEval(
	ctx context.Context, req _EvalOcrCtpnReq, env _EvalEnv,
) (_EvalOcrCtpnResp, error) {
	var (
		url    = ot.host + "/v1/eval/ocr-ctpn"
		client = newRPCClient(env, ot.timeout)
		resp   _EvalOcrCtpnResp
	)
	err := callRetry(ctx,
		func(ctx context.Context) error {
			return client.CallWithJson(ctx, &resp, "POST", url, &req)
		})
	return resp, err
}

func (ot _OcrText) RecognizeEval(
	ctx context.Context, req _EvalOcrTextRecognizeReq, env _EvalEnv,
) (_EvalOcrTextRecognizeResp, error) {
	var (
		url    = ot.host + "/v1/eval/ocr-recognize"
		client = newRPCClient(env, ot.timeout)
		resp   _EvalOcrTextRecognizeResp
	)
	err := callRetry(ctx,
		func(ctx context.Context) error {
			return client.CallWithJson(ctx, &resp, "POST", url, &req)
		})
	return resp, err
}

func (ot _OcrText) SceneDetectEval(
	ctx context.Context, req _EvalOcrSceneDetectReq, env _EvalEnv,
) (_EvalOcrSceneDetectResp, error) {
	var (
		url    = ot.host + "/v1/eval/ocr-scene-detect"
		client = newRPCClient(env, ot.timeout)
		resp   _EvalOcrSceneDetectResp
	)
	err := callRetry(ctx,
		func(ctx context.Context) error {
			return client.CallWithJson(ctx, &resp, "POST", url, &req)
		})
	return resp, err
}

func (ot _OcrText) SceneRecognizeEval(
	ctx context.Context, req _EvalOcrSceneRecognizeReq, env _EvalEnv,
) (_EvalOcrSceneRecognizeResp, error) {
	var (
		url    = ot.host + "/v1/eval/ocr-scene-recog"
		client = newRPCClient(env, ot.timeout)
		resp   _EvalOcrSceneRecognizeResp
	)
	err := callRetry(ctx,
		func(ctx context.Context) error {
			return client.CallWithJson(ctx, &resp, "POST", url, &req)
		})
	return resp, err
}

func (s *Service) PostOcrText(ctx context.Context, args *OcrTextReq, env *authstub.Env) (ret *OcrTextResp, err error) {
	var (
		uid     = env.UserInfo.Uid
		utype   = env.UserInfo.Utype
		evalEnv = _EvalEnv{Uid: uid, Utype: utype}
	)
	ret = &OcrTextResp{}
	ret.Result.Bboxes = make([][4][2]int, 0)
	ret.Result.Texts = make([]string, 0)
	ctx, xl := ctxAndLog(ctx, env.W, env.Req)

	if strings.TrimSpace(args.Data.URI) == "" {
		xl.Error("empty data.uri")
		return nil, ErrArgs
	}

	// ocr text classify
	var otClassifyReq _EvalOcrTextClassifyReq
	otClassifyReq.Data.URI = args.Data.URI
	otClassifyResp, err := s.iOcrText.ClassifyEval(ctx, otClassifyReq, evalEnv)
	if err != nil {
		xl.Errorf("image ocr text classify failed. %v", err)
		return
	}
	if len(otClassifyResp.Result.Confidences) != 1 {
		xl.Errorf("ocr text classify expect to get one result, but got %d", len(otClassifyResp.Result.Confidences))
		return nil, ErrArgs
	}
	ret.Result.Type = otClassifyResp.Result.Confidences[0].Class
	xl.Info("ocr text image_type: ", ret.Result.Type)

	if otClassifyResp.Result.Confidences[0].Class != OtherTextScene && otClassifyResp.Result.Confidences[0].Class != IgnoreImageType {
		// weixin-weibo process
		// ocr text detect
		var (
			otDetectReq  _EvalOcrCtpnReq
			otDetectResp _EvalOcrCtpnResp
		)
		otDetectReq.Data.URI = args.Data.URI
		otDetectResp, err = s.iOcrText.DetectEval(ctx, otDetectReq, evalEnv)
		if err != nil {
			xl.Errorf("image ocr text detect failed. %v", err)
			return
		}
		if len(otDetectResp.Result.Bboxes) == 0 {
			xl.Debugf("ocr text detect got no bbox, ignore this image")
			return
		}
		xl.Infof("ocr text detect got bboxes: %v", otDetectResp.Result.Bboxes)

		// ocr text recognize
		var (
			otRecognizeReq  _EvalOcrTextRecognizeReq
			otRecognizeResp _EvalOcrTextRecognizeResp
		)
		otRecognizeReq.Data.URI = args.Data.URI
		for i := 0; i < len(otDetectResp.Result.Bboxes); i++ {
			otRecognizeReq.Params.Bboxes = append(otRecognizeReq.Params.Bboxes, [4]int{
				otDetectResp.Result.Bboxes[i][0][0],
				otDetectResp.Result.Bboxes[i][0][1],
				otDetectResp.Result.Bboxes[i][2][0],
				otDetectResp.Result.Bboxes[i][2][1],
			})
		}
		otRecognizeReq.Params.ImageType = ret.Result.Type
		otRecognizeResp, err = s.iOcrText.RecognizeEval(ctx, otRecognizeReq, evalEnv)
		if err != nil {
			xl.Errorf("image ocr text recognize failed. %v", err)
			return
		}
		ret.Result.Texts = append(ret.Result.Texts, otRecognizeResp.Result.Texts...)
		ret.Result.Bboxes = append(ret.Result.Bboxes, otRecognizeResp.Result.Bboxes...)
		xl.Infof("ocr text recognize got text: %v", ret.Result.Texts)
	} else {
		// scene process
		// ocr text detect
		var (
			otDetectReq  _EvalOcrSceneDetectReq
			otDetectResp _EvalOcrSceneDetectResp
		)
		otDetectReq.Data.URI = args.Data.URI
		otDetectResp, err = s.iOcrText.SceneDetectEval(ctx, otDetectReq, evalEnv)
		if err != nil {
			xl.Errorf("image ocr text detect failed. %v", err)
			return
		}
		if len(otDetectResp.Result.Bboxes) == 0 {
			xl.Debugf("ocr scene detect got not bbox, ignore this image")
			return
		}
		for i := 0; i < len(otDetectResp.Result.Bboxes); i++ {
			ret.Result.Bboxes = append(ret.Result.Bboxes, [4][2]int{
				{otDetectResp.Result.Bboxes[i][0], otDetectResp.Result.Bboxes[i][1]},
				{otDetectResp.Result.Bboxes[i][2], otDetectResp.Result.Bboxes[i][3]},
				{otDetectResp.Result.Bboxes[i][4], otDetectResp.Result.Bboxes[i][5]},
				{otDetectResp.Result.Bboxes[i][6], otDetectResp.Result.Bboxes[i][7]}})
		}

		xl.Infof("ocr scene detect got bboxes: %v", ret.Result.Bboxes)

		// ocr text recognize
		var (
			otRecognizeReq  _EvalOcrSceneRecognizeReq
			otRecognizeResp _EvalOcrSceneRecognizeResp
		)
		otRecognizeReq.Data.URI = args.Data.URI
		otRecognizeReq.Params.Bboxes = otDetectResp.Result.Bboxes
		otRecognizeResp, err = s.iOcrText.SceneRecognizeEval(ctx, otRecognizeReq, evalEnv)
		if err != nil {
			xl.Errorf("image ocr text recognize failed. %v", err)
			return
		}
		xl.Infof("ocr text recognize got text: %v", otRecognizeResp.Result.Texts)
		for i := 0; i < len(otRecognizeResp.Result.Texts); i++ {
			ret.Result.Texts = append(ret.Result.Texts, otRecognizeResp.Result.Texts[i].Text)
		}

		xl.Infof("ocr text recognize got text: %v", ret.Result.Texts)
	}

	return
}
