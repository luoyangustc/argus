package image_sync

import (
	"context"

	"github.com/go-kit/kit/endpoint"
	"qiniu.com/argus/service/middleware"
	sbiz "qiniu.com/argus/service/scenario/biz"
	scenario "qiniu.com/argus/service/scenario/image_sync"
	"qiniu.com/argus/service/service/biz"
	pimage "qiniu.com/argus/service/service/image"
	"qiniu.com/argus/service/service/image/ocrtext"
	"qiniu.com/argus/service/transport"
)

const (
	VERSION = "1.0.0"
)

// eval 统一名称
const (
	eocName  = "ocr-classify"
	eodName  = "ocr-ctpn"
	eorName  = "ocr-recognize"
	eosdName = "ocr-scene-detect"
	eosrName = "ocr-scene-recog"
)

var (
	_DOC_OCRTEXT = scenario.APIDoc{
		Version: VERSION,
		Desc:    []string{"文本信息识别"},
		Request: `POST /v1/ocr/text  Http/1.1
Content-Type:application/json
{
	"data":{
		"uri": "http://oqgascup5.bkt.clouddn.com/ocr/WechatIMG61.png"
	}
}`,
		Response: `200 ok
Content-Type:application/json
{
	"code":0,
	"message":"",
	"result":{
		"type": "wechat",
		"bboxes": [
			[[140,1125],[596,1161],[140,1046],[331,1082]],
			[[140,1005],[563,1041],[141,1167],[238,1200]],
			[[140,924],[594,962],[141,237],[605,273]],
			...,
			[[119,182],[194,210],[119,502],[194,531]]
		],
		"texts": [
			'防治疗中有种非医果药物做贝',
			'抗。2万多一计副作册小发',
			'就开口了，跟勿说了你前天的瘤',
			...,
			'手术了，化疗吧。'
		]
	}
}`,
		RequestParam: []scenario.APIDocParam{
			{Name: "uri", Type: "string", Desc: "图片资源地址"},
		},
		ResponseParam: []scenario.APIDocParam{
			{Name: "code", Type: "int", Desc: "0:表示处理成功；不为0:表示出错"},
			{Name: "message", Type: "string", Desc: "描述结果或出错信息"},
			{Name: "type", Type: "string", Desc: "文本类别，{'wechat','blog','other-text','normal'}，分别表示微信、微博、其他文本、非文本"},
			{Name: "bboxes", Type: "list[4,2]", Desc: "返回的图片中所有的文本框位置，为顺时针/逆时针方向旋转的任意四边形[[x1,y1],[x2,y2],[x3,y3],[x4,y4]]"},
			{Name: "texts", Type: "list", Desc: "对应四边形框中的文字"},
		},
		ErrorMessage: []scenario.APIDocError{
			{Code: 4000801, Desc: "未检测到文本信息"},
		},
	}
)

func Import(serviceID string) func(interface{}) {
	return func(is interface{}) {
		Init(is.(scenario.ImageServer), serviceID)
	}
}

func Init(is scenario.ImageServer, serviceID string) {
	var eocSet sbiz.ServiceEvalSetter
	var eodSet sbiz.ServiceEvalSetter
	var eorSet sbiz.ServiceEvalSetter
	var eosdSet sbiz.ServiceEvalSetter
	var eosrSet sbiz.ServiceEvalSetter

	var eoc ocrtext.EvalOcrTextClassifyService
	var eod ocrtext.EvalOcrCtpnService
	var eor ocrtext.EvalOcrTextRecognizeService
	var eosd ocrtext.EvalOcrSceneDetectService
	var eosr ocrtext.EvalOcrSceneRecognizeService

	set := is.Register(
		scenario.ServiceInfo{ID: serviceID, Name: "ocrtext", Version: VERSION},
		&middleware.ServiceFactory{
			New: func() middleware.Service {
				{
					eoc1, _ := eocSet.Gen()
					eoc = eoc1.(ocrtext.EvalOcrTextClassifyService)

					eod1, _ := eodSet.Gen()
					eod = eod1.(ocrtext.EvalOcrCtpnService)

					eor1, _ := eorSet.Gen()
					eor = eor1.(ocrtext.EvalOcrTextRecognizeService)

					eosd1, _ := eosdSet.Gen()
					eosd = eosd1.(ocrtext.EvalOcrSceneDetectService)

					eosr1, _ := eosrSet.Gen()
					eosr = eosr1.(ocrtext.EvalOcrSceneRecognizeService)

				}
				s, _ := ocrtext.NewOcrTextService(eoc, eod, eor, eosd, eosr)
				return s
			},
			NewShell: func() middleware.ServiceEndpoints { return ocrtext.OcrTextEndpoints{} },
		}).
		UpdateRouter(func(
			sf func() middleware.Service,
			imageParser func() pimage.IImageParse,
			path func(string) *scenario.ServiceRoute,
		) error {
			endp := func() ocrtext.OcrTextEndpoints {
				svc := sf()
				endp, ok := svc.(ocrtext.OcrTextEndpoints)
				if !ok {
					svc, _ = middleware.MakeMiddleware(svc, ocrtext.OcrTextEndpoints{}, nil, nil)
					endp = svc.(ocrtext.OcrTextEndpoints)
				}
				return endp
			}

			type Req struct {
				Data struct {
					URI string `json:"uri"`
				} `json:"data"`
			}

			path("/v1/ocr/text").Doc(_DOC_OCRTEXT).Route().
				Methods("POST").Handler(transport.MakeHttpServer(
				func(ctx context.Context, req0 interface{}) (interface{}, error) {
					req1, _ := req0.(Req)
					var req2 ocrtext.OcrTextReq
					var err error
					req2.Data.IMG, err = imageParser().ParseImage(ctx, req1.Data.URI)
					if err != nil {
						return nil, err
					}
					return endp().OcrTextEP(ctx, req2)
				},
				Req{}))
			return nil
		})

	eocSet = set.NewEval(
		sbiz.ServiceEvalInfo{Name: eocName, Version: "1.0.0"},
		OcrTextClassifyEvalClient,
		func() middleware.ServiceEndpoints { return ocrtext.EvalOcrTextClassifyEndpoints{} },
	).SetModel(biz.EvalModelConfig{
		Image: "hub2.qiniu.com/1381102897/ava-eval-ataraxia-ocr-weixin-weibo-text-classification:201806061829--201806061842-v153-dev",
		Model: "ava-ocr-text-classify/caffe-classify/ocr-classification-20180605.tar",
		Args: &biz.ModelConfigArgs{
			BatchSize:  1,
			ImageWidth: 225,
			CustomValues: map[string]interface{}{
				"threshold":        0.6,
				"mean_value":       []float32{103.94, 116.78, 123.68},
				"input_scale":      0.017,
				"other_class":      "normal",
				"resize_crop_size": []int{256, 256, 225, 225},
			},
		},
		Type: biz.EvalRunTypeServing,
	})

	eodSet = set.NewEval(
		sbiz.ServiceEvalInfo{Name: eodName, Version: "1.0.0"},
		OcrTextCtpnEvalClient,
		func() middleware.ServiceEndpoints { return ocrtext.EvalOcrCtpnEndpoints{} },
	).SetModel(biz.EvalModelConfig{
		Image: "hub2.qiniu.com/1381102897/ava-eval-ataraxia-ocr-detection-ctpn:2018090801--201809081040-v362-dev",
		Model: "ava-ocr-detect/ocr-ctpn-model-20180816.tar",
		Args: &biz.ModelConfigArgs{
			BatchSize:    1,
			ImageWidth:   225,
			CustomValues: map[string]interface{}{},
		},
		Type: biz.EvalRunTypeServing,
	})

	eorSet = set.NewEval(
		sbiz.ServiceEvalInfo{Name: eorName, Version: "1.0.0"},
		OcrTextRecognizeEvalClient,
		func() middleware.ServiceEndpoints { return ocrtext.EvalOcrTextRecognizeEndpoints{} },
	).SetModel(biz.EvalModelConfig{
		Image: "hub2.qiniu.com/1381102897/ava-eval-ataraxia-ocr-weixin-weibo-text-recognition:2018072401--201807241314-v282-dev",
		Model: "ava-ocr-text-recognize/other-ocr/ocr-recognition-201802021114.tar",
		Args: &biz.ModelConfigArgs{
			BatchSize:    1,
			ImageWidth:   225,
			CustomValues: map[string]interface{}{},
		},
		Type: biz.EvalRunTypeServing,
	})

	eosdSet = set.NewEval(
		sbiz.ServiceEvalInfo{Name: eosdName, Version: "1.0.0"},
		OcrSceneDetectEvalClient,
		func() middleware.ServiceEndpoints { return ocrtext.EvalOcrSceneDetectEndpoints{} },
	).SetModel(biz.EvalModelConfig{
		Image: "hub2.qiniu.com/1381102897/ava-eval-ataraxia-ocr-scene-detection:201808101422--201808101511-v344-dev",
		Model: "ava-ocr-scene-detection/ocr-detect-east-20180809.tar",
		Args: &biz.ModelConfigArgs{
			BatchSize:    1,
			ImageWidth:   225,
			CustomValues: map[string]interface{}{},
		},
		Type: biz.EvalRunTypeServing,
	})

	eosrSet = set.NewEval(
		sbiz.ServiceEvalInfo{Name: eosrName, Version: "1.0.0"},
		OcrSceneRecogEvalClient,
		func() middleware.ServiceEndpoints { return ocrtext.EvalOcrSceneRecognizeEndpoints{} },
	).SetModel(biz.EvalModelConfig{
		Image: "hub2.qiniu.com/1381102897/ava-eval-ataraxia-ocr-scene-recognition:1457",
		Model: "ava-ocr-scene-recognition/ocr-scene-recognition-20180513.tar",
		Args: &biz.ModelConfigArgs{
			BatchSize:    1,
			ImageWidth:   225,
			CustomValues: map[string]interface{}{},
		},
		Type: biz.EvalRunTypeServing,
	})

	// 添加一种4卡的部署方式
	set.AddEvalsDeployModeOnGPU("", [][]sbiz.ServiceEvalDeployProcess{
		{
			{Name: eodName, Num: 1}, // ctpn 独占 GPU0
		},
		{
			{Name: eosdName, Num: 1}, // sence detect 独占 GPU1
		},
		{
			{Name: eocName, Num: 1}, // 剩下的两张卡部署在剩下的三个原子服务
			{Name: eorName, Num: 1},
			{Name: eosrName, Num: 1},
		},
		{
			{Name: eocName, Num: 1},
			{Name: eorName, Num: 1},
			{Name: eosrName, Num: 1},
		},
	})
}

type Req struct {
	Data struct {
		URI string `json:"uri"`
	} `json:"data"`
}

func OcrTextClassifyEvalClient(
	client func(method string, respSample interface{}) (endpoint.Endpoint, error),
) middleware.Service {
	end, _ := client("POST", ocrtext.EvalOcrTextClassifyResp{})
	return ocrtext.EvalOcrTextClassifyEndpoints{
		EvalOcrTextClassifyEP: func(ctx context.Context, req0 interface{}) (interface{}, error) {
			req1, _ := req0.(ocrtext.EvalOcrTextClassifyReq)
			var req2 Req
			req2.Data.URI = string(req1.Data.IMG.URI)
			return end(ctx, req2)
		},
	}
}

// type ReqCtpn struct {
// 	Data struct {
// 		URI string `json:"uri"`
// 	} `json:"data"`
// }

func OcrTextCtpnEvalClient(
	client func(method string, respSample interface{}) (endpoint.Endpoint, error),
) middleware.Service {
	end, _ := client("POST", ocrtext.EvalOcrCtpnResp{})
	return ocrtext.EvalOcrCtpnEndpoints{
		EvalOcrCtpnEP: func(ctx context.Context, req0 interface{}) (interface{}, error) {
			req1, _ := req0.(ocrtext.EvalOcrCtpnReq)
			var req2 Req
			req2.Data.URI = string(req1.Data.IMG.URI)
			return end(ctx, req2)
		},
	}
}

type ReqRecognize struct {
	Data struct {
		URI string `json:"uri"`
	} `json:"data"`
	Params struct {
		Bboxes    [][4]int `json:"bboxes"`
		ImageType string   `json:"image_type"`
	} `json:"params"`
}

func OcrTextRecognizeEvalClient(
	client func(method string, respSample interface{}) (endpoint.Endpoint, error),
) middleware.Service {
	end, _ := client("POST", ocrtext.EvalOcrTextRecognizeResp{})
	return ocrtext.EvalOcrTextRecognizeEndpoints{
		EvalOcrTextRecognizeEP: func(ctx context.Context, req0 interface{}) (interface{}, error) {
			req1, _ := req0.(ocrtext.EvalOcrTextRecognizeReq)
			var req2 ReqRecognize
			req2.Data.URI = string(req1.Data.IMG.URI)
			req2.Params = req1.Params
			return end(ctx, req2)
		},
	}
}

// type ReqScene struct {
// 	Data struct {
// 		URI string `json:"uri"`
// 	} `json:"data"`
// }

func OcrSceneDetectEvalClient(
	client func(method string, respSample interface{}) (endpoint.Endpoint, error),
) middleware.Service {
	end, _ := client("POST", ocrtext.EvalOcrSceneDetectResp{})
	return ocrtext.EvalOcrSceneDetectEndpoints{
		EvalOcrSceneDetectEP: func(ctx context.Context, req0 interface{}) (interface{}, error) {
			req1, _ := req0.(ocrtext.EvalOcrSceneDetectReq)
			var req2 Req
			req2.Data.URI = string(req1.Data.IMG.URI)
			return end(ctx, req2)
		},
	}
}

type ReqSceneRecognize struct {
	Data struct {
		URI string `json:"uri"`
	} `json:"data"`
	Params struct {
		Bboxes [][8]int `json:"bboxes"`
	} `json:"params"`
}

func OcrSceneRecogEvalClient(
	client func(method string, respSample interface{}) (endpoint.Endpoint, error),
) middleware.Service {
	end, _ := client("POST", ocrtext.EvalOcrSceneRecognizeResp{})
	return ocrtext.EvalOcrSceneRecognizeEndpoints{
		EvalOcrSceneRecognizeEP: func(ctx context.Context, req0 interface{}) (interface{}, error) {
			req1, _ := req0.(ocrtext.EvalOcrSceneRecognizeReq)
			var req2 ReqSceneRecognize
			req2.Data.URI = string(req1.Data.IMG.URI)
			req2.Params = req1.Params
			return end(ctx, req2)
		},
	}
}
