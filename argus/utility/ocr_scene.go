package utility

import (
	"context"
	"strings"
	"time"

	"qiniu.com/auth/authstub.v1"
)

type OcrSceneResp struct {
	Code    int            `json:"code"`
	Message string         `json:"message"`
	Result  OcrSceneResult `json:"result"`
}

type OcrSceneResult struct {
	Bboxes [][8]int `json:"bboxes"`
	Text   []string `json:"text"`
}

type iOcrScene interface {
	SceneDetectEval(context.Context, _EvalOcrSceneDetectReq, _EvalEnv) (_EvalOcrSceneDetectResp, error)
	SceneRecognizeEval(context.Context, _EvalOcrSceneRecognizeReq, _EvalEnv) (_EvalOcrSceneRecognizeResp, error)
}

type _OcrScene struct {
	host    string
	timeout time.Duration
}

func newOcrScene(host string, timeout time.Duration) iOcrScene {
	return _OcrScene{host: host, timeout: timeout}
}

func (ot _OcrScene) SceneDetectEval(
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

func (ot _OcrScene) SceneRecognizeEval(
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

func (s *Service) PostOcrScene(ctx context.Context, args *OcrTextReq, env *authstub.Env) (ret *OcrSceneResp, err error) {
	var (
		uid     = env.UserInfo.Uid
		utype   = env.UserInfo.Utype
		evalEnv = _EvalEnv{Uid: uid, Utype: utype}
	)
	ret = &OcrSceneResp{}
	ret.Result.Bboxes = make([][8]int, 0)
	ret.Result.Text = make([]string, 0)
	ctx, xl := ctxAndLog(ctx, env.W, env.Req)

	if strings.TrimSpace(args.Data.URI) == "" {
		xl.Error("empty data.uri")
		return nil, ErrArgs
	}

	// ocr text detect
	var (
		otDetectReq  _EvalOcrSceneDetectReq
		otDetectResp _EvalOcrSceneDetectResp
	)
	otDetectReq.Data.URI = args.Data.URI
	otDetectResp, err = s.iOcrScene.SceneDetectEval(ctx, otDetectReq, evalEnv)
	if err != nil {
		xl.Errorf("image ocr scene detect failed. %v", err)
		return
	}
	if len(otDetectResp.Result.Bboxes) == 0 {
		xl.Debugf("ocr scene detect got not bbox, ignore this image")
		return
	}
	ret.Result.Bboxes = append(ret.Result.Bboxes, otDetectResp.Result.Bboxes...)
	xl.Infof("ocr scene detect got bboxes: %v", ret.Result.Bboxes)

	// ocr text recognize
	var (
		otRecognizeReq  _EvalOcrSceneRecognizeReq
		otRecognizeResp _EvalOcrSceneRecognizeResp
	)
	otRecognizeReq.Data.URI = args.Data.URI
	otRecognizeReq.Params.Bboxes = ret.Result.Bboxes
	otRecognizeResp, err = s.iOcrScene.SceneRecognizeEval(ctx, otRecognizeReq, evalEnv)
	if err != nil {
		xl.Errorf("image ocr text recognize failed. %v", err)
		return
	}
	xl.Infof("ocr text recognize got text: %v", otRecognizeResp.Result.Texts)
	for i := 0; i < len(otRecognizeResp.Result.Texts); i++ {
		ret.Result.Text = append(ret.Result.Text, otRecognizeResp.Result.Texts[i].Text)
	}

	xl.Infof("ocr text recognize got text: %v", ret.Result.Text)

	return
}
