package image_sync

import (
	"context"

	"github.com/go-kit/kit/endpoint"
	"qiniu.com/argus/service/middleware"
	sbiz "qiniu.com/argus/service/scenario/biz"
	scenario "qiniu.com/argus/service/scenario/image_sync"
	"qiniu.com/argus/service/service/biz"
	pimage "qiniu.com/argus/service/service/image"
	"qiniu.com/argus/service/service/image/ocrbankcard"
	"qiniu.com/argus/service/transport"
)

const (
	VERSION = "1.0.0"
)

// eval 统一名称
const (
	eocName  = "ocr-ctpn"
	eoscName = "ocr-sari-crann"
)

var (
	_DOC_OCRBANKCARD = scenario.APIDoc{
		Version: VERSION,
		Desc:    []string{"文本识别信息"},
		Request: `POST /v1/ocr/bankcard Http/1.1
Content-Type:application/json
{
	"data": {
		"uri": "http://p9zv90cqq.bkt.clouddn.com/001.jpg"
	}
}`,
		Response: `200 ok
Content-Type:application/json
{
	"code": 0,
	"message": "",
	"result": {
		"bboxes": [
			[[134,227],[419,227],[419,262],[134,262]],
			[[115,50],[115,100],[232,100],[232,50]]
		],
		"res": {
			"卡号": "9558801001128579882", 
			"开户银行": "邑)中国工商银行"
		}
	}
}`,

		RequestParam: []scenario.APIDocParam{
			{Name: "uri", Type: "string", Desc: "图片资源地址"},
		},
		ResponseParam: []scenario.APIDocParam{
			{Name: "code", Type: "int", Desc: "0:表示处理成功；不为0:表示出错"},
			{Name: "message", Type: "string", Desc: "描述结果或出错信息"},
			{Name: "bboxes", Type: "list[4,2]", Desc: "返回的图片中卡号和开户行的文本框位置，为顺时针方向旋转的任意四边形[[x1,y1],[x2,y2],[x3,y3],[x4,y4]],若没找到则返回[[0,0],[0,0],[0,0],[0,0]]"},
			{Name: "res", Type: "map[string]string", Desc: "识别后信息结构化的结果"},
			{Name: "res[卡号]", Type: "string", Desc: "卡号,若没有识别出此字段则为空"},
			{Name: "res[开户银行]", Type: "string", Desc: "开户银行，若没有识别出此字段则为空"},
		},
		ErrorMessage: []scenario.APIDocError{
			{Code: 4000803, Desc: "未检测到银行卡信息"},
		},
	}
)

func Import(serviceID string) func(interface{}) {
	return func(is interface{}) {
		Init(is.(scenario.ImageServer), serviceID)
	}
}

type ReqDetect struct {
	Data struct {
		URI string `json:"uri"`
	} `json:"data"`
}

type ReqRecog struct {
	Data struct {
		URI string `json:"uri"`
	} `json:"data"`
	Params struct {
		Bboxes [][4][2]int `json:"bboxes"`
	} `json:"params"`
}

func Init(is scenario.ImageServer, serviceID string) {
	var eodSet sbiz.ServiceEvalSetter
	var eorSet sbiz.ServiceEvalSetter

	var eod ocrbankcard.EvalOcrSariBankcardDetectService
	var eor ocrbankcard.EvalOcrSariBankcardRecogService

	set := is.Register(
		scenario.ServiceInfo{ID: serviceID, Name: "ocrbankcard", Version: VERSION},
		&middleware.ServiceFactory{
			New: func() middleware.Service {
				{
					eod1, _ := eodSet.Gen()
					eod = eod1.(ocrbankcard.EvalOcrSariBankcardDetectService)

					eor1, _ := eorSet.Gen()
					eor = eor1.(ocrbankcard.EvalOcrSariBankcardRecogService)
				}
				s, _ := ocrbankcard.NewOcrBankcardService(eod, eor)
				return s
			},
			NewShell: func() middleware.ServiceEndpoints { return ocrbankcard.OcrBankcardEndpoints{} },
		}).
		UpdateRouter(func(
			sf func() middleware.Service,
			imageParser func() pimage.IImageParse,
			path func(string) *scenario.ServiceRoute,
		) error {
			endp := func() ocrbankcard.OcrBankcardEndpoints {
				svc := sf()
				endp, ok := svc.(ocrbankcard.OcrBankcardEndpoints)
				if !ok {
					svc, _ = middleware.MakeMiddleware(svc, ocrbankcard.OcrBankcardEndpoints{}, nil, nil)
					endp = svc.(ocrbankcard.OcrBankcardEndpoints)
				}
				return endp
			}

			type Req struct {
				Data struct {
					URI string `json:"uri"`
				} `json:"data"`
			}

			path("/v1/ocr/bankcard").Doc(_DOC_OCRBANKCARD).Route().
				Methods("POST").Handler(transport.MakeHttpServer(
				func(ctx context.Context, req0 interface{}) (interface{}, error) {
					req1, _ := req0.(Req)
					var req2 ocrbankcard.OcrSariBankcardReq
					var err error
					req2.Data.IMG, err = imageParser().ParseImage(ctx, req1.Data.URI)
					if err != nil {
						return nil, err
					}
					return endp().OcrBankcardEP(ctx, req2)
				},
				Req{}))
			return nil
		})

	eodSet = set.NewEval(
		sbiz.ServiceEvalInfo{Name: "ocr-ctpn", Version: "1.0.0"},
		OcrSariBankcardDetectEvalClient,
		func() middleware.ServiceEndpoints { return ocrbankcard.EvalOcrSariBankcardDetectEndpoints{} },
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
		sbiz.ServiceEvalInfo{Name: "ocr-sari-crann", Version: "1.0.0"},
		OcrSariBankcardRecognizeEvalClient,
		func() middleware.ServiceEndpoints { return ocrbankcard.EvalOcrSariBankcardRecogEndpoints{} },
	).SetModel(biz.EvalModelConfig{
		Image: "hub2.qiniu.com/1381102897/ava-eval-ataraxia-ocr-sari-crann:201808051919--201808052143-v336-dev",
		Model: "ava-ocr-sari-crann/ocr-sari-crann-20180607.tar",
		Args: &biz.ModelConfigArgs{
			BatchSize:    1,
			ImageWidth:   225,
			CustomValues: map[string]interface{}{},
		},
		Type: biz.EvalRunTypeServing,
	})

	// 添加一种2卡的部署方式
	set.AddEvalsDeployModeOnGPU("", [][]sbiz.ServiceEvalDeployProcess{
		{
			{Name: eocName, Num: 1}, // ctpn 独占 GPU0
		},
		{
			{Name: eoscName, Num: 1},
		},
	})
}

func OcrSariBankcardDetectEvalClient(
	client func(method string, respSample interface{}) (endpoint.Endpoint, error),
) middleware.Service {
	end, _ := client("POST", ocrbankcard.EvalOcrSariBankcardDetectResp{})
	return ocrbankcard.EvalOcrSariBankcardDetectEndpoints{
		EvalOcrSariBankcardDetectEP: func(ctx context.Context, req0 interface{}) (interface{}, error) {
			req1, _ := req0.(ocrbankcard.EvalOcrSariBankcardDetectReq)
			var req2 ReqDetect
			req2.Data.URI = string(req1.Data.IMG.URI)
			return end(ctx, req2)
		},
	}
}

func OcrSariBankcardRecognizeEvalClient(
	client func(method string, respSample interface{}) (endpoint.Endpoint, error),
) middleware.Service {
	end, _ := client("POST", ocrbankcard.EvalOcrSariBankcardRecogResp{})
	return ocrbankcard.EvalOcrSariBankcardRecogEndpoints{
		EvalOcrSariBankcardRecogEP: func(ctx context.Context, req0 interface{}) (interface{}, error) {
			req1, _ := req0.(ocrbankcard.EvalOcrSariBankcardRecogReq)
			var req2 ReqRecog
			req2.Data.URI = string(req1.Data.IMG.URI)
			req2.Params = req1.Params
			return end(ctx, req2)
		},
	}
}
