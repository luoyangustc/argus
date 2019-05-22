package image_sync

import (
	"context"

	"github.com/go-kit/kit/endpoint"

	"qiniu.com/argus/service/middleware"
	sbiz "qiniu.com/argus/service/scenario/biz"
	scenario "qiniu.com/argus/service/scenario/image_sync"
	"qiniu.com/argus/service/service/biz"
	pimage "qiniu.com/argus/service/service/image"
	"qiniu.com/argus/service/service/image/ocridcard"
	"qiniu.com/argus/service/transport"
)

const (
	VERSION = "1.0.0"
)

const (
	eosdName = "ocr-scene-detect"
	eoscName = "ocr-sari-crann"
)

var (
	_DOC_OCRIDCARD = scenario.APIDoc{
		Version: VERSION,
		Desc:    []string{"身份证信息识别"},
		Request: `POST /v1/ocr/idcard Http/1.1
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
		"uri": "data:application/octet-stream;base64, ...",
		"bboxes": [
			[[134,227],[419,227],[419,262],[134,262]],
			...
			[[115,50],[115,100],[232,100],[232,50]]
		],
		"type": 0,
		"res": {
			"住址": "河南省项城市芙蓉巷东四胡同2号",
			"公民身份号码": "412702199705127504",
			"出生": "1997年5月12日",
			"姓名": "张杰",
			"性别": "女",
			"民族": "汉"
		}
	}
}`,
		RequestParam: []scenario.APIDocParam{
			{Name: "uri", Type: "string", Desc: "图片资源地址"},
		},
		ResponseParam: []scenario.APIDocParam{
			{Name: "code", Type: "int", Desc: "0:表示处理成功；不为0:表示出错"},
			{Name: "message", Type: "string", Desc: "描述结果或出错信息"},
			{Name: "uri", Type: "string", Desc: "截取原图中身份证区域后的图片 base64 编码"},
			{Name: "bboxes", Type: "list[4,2]", Desc: "返回的图片中所有的文本框位置，为顺时针/逆时针方向旋转的任意四边形[[x1,y1],[x2,y2],[x3,y3],[x4,y4]]"},
			{Name: "type", Type: "int", Desc: "身份证正反面信息，0:正面，1:背面"},
			{Name: "res[姓名]", Type: "string", Desc: "姓名"},
			{Name: "res[性别]", Type: "string", Desc: "性别"},
			{Name: "res[民族]", Type: "string", Desc: "民族"},
			{Name: "res[出生]", Type: "string", Desc: "出生"},
			{Name: "res[住址]", Type: "string", Desc: "住址"},
			{Name: "res[公民身份号码]", Type: "string", Desc: "公民身份号码"},
			{Name: "res[有效期限]", Type: "string", Desc: "有效期限"},
			{Name: "res[签发机关]", Type: "string", Desc: "签发机关"},
		},
		ErrorMessage: []scenario.APIDocError{
			{Code: 4000802, Desc: "未检测到身份证信息"},
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

type ReqPre struct {
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

func Init(is scenario.ImageServer, serviceID string) {
	var eodSet sbiz.ServiceEvalSetter
	var eorSet sbiz.ServiceEvalSetter
	var eopSet sbiz.ServiceEvalSetter

	var eod ocridcard.EvalOcrSariIdcardDetectService
	var eor ocridcard.EvalOcrSariIdcardRecogService
	var eop ocridcard.EvalOcrSariIdcardPreProcessService

	set := is.Register(
		scenario.ServiceInfo{ID: serviceID, Name: "ocridcard", Version: VERSION},
		&middleware.ServiceFactory{
			New: func() middleware.Service {
				{
					eod1, _ := eodSet.Gen()
					eod = eod1.(ocridcard.EvalOcrSariIdcardDetectService)

					eor1, _ := eorSet.Gen()
					eor = eor1.(ocridcard.EvalOcrSariIdcardRecogService)

					eop1, _ := eopSet.Gen()
					eop = eop1.(ocridcard.EvalOcrSariIdcardPreProcessService)
				}
				s, _ := ocridcard.NewOcrIdcardService(eod, eor, eop)
				return s
			},
			NewShell: func() middleware.ServiceEndpoints { return ocridcard.OcrIdcardEndpoints{} },
		}).
		UpdateRouter(func(
			sf func() middleware.Service,
			imageParser func() pimage.IImageParse,
			path func(string) *scenario.ServiceRoute,
		) error {
			endp := func() ocridcard.OcrIdcardEndpoints {
				svc := sf()
				endp, ok := svc.(ocridcard.OcrIdcardEndpoints)
				if !ok {
					svc, _ = middleware.MakeMiddleware(svc, ocridcard.OcrIdcardEndpoints{}, nil, nil)
					endp = svc.(ocridcard.OcrIdcardEndpoints)
				}
				return endp
			}

			type Req struct {
				Data struct {
					URI string `json:"uri"`
				} `json:"data"`
			}

			path("/v1/ocr/idcard").Doc(_DOC_OCRIDCARD).Route().
				Methods("POST").Handler(transport.MakeHttpServer(
				func(ctx context.Context, req0 interface{}) (interface{}, error) {
					req1, _ := req0.(Req)
					var req2 ocridcard.OcrSariIdcardReq
					// req2.Data.IMG = req1.Data.IMG
					var err error
					req2.Data.IMG, err = imageParser().ParseImage(ctx, req1.Data.URI)
					if err != nil {
						return nil, err
					}
					return endp().OcrIdcardEP(ctx, req2)
				},
				Req{}))
			return nil
		})

	eodSet = set.NewEval(
		sbiz.ServiceEvalInfo{Name: "ocr-scene-detect", Version: "1.0.0"},
		OcrSariIdcardDetectEvalClient,
		func() middleware.ServiceEndpoints { return ocridcard.EvalOcrSariIdcardDetectEndpoints{} },
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

	eorSet = set.NewEval(
		sbiz.ServiceEvalInfo{Name: "ocr-sari-crann", Version: "1.0.0"},
		OcrSariIdcardRecogEvalClient,
		func() middleware.ServiceEndpoints { return ocridcard.EvalOcrSariIdcardRecogEndpoints{} },
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

	eopSet = set.NewEval(
		sbiz.ServiceEvalInfo{Name: "ocr-sari-id-pre", Version: "1.0.0"},
		OcrSariIdcardPreEvalClient,
		func() middleware.ServiceEndpoints { return ocridcard.EvalOcrSariIdcardPreProcessEndpoints{} },
	).SetModel(biz.EvalModelConfig{
		Image: "hub2.qiniu.com/1381102897/ava-eval-ataraxia-ocr-sari-idcard-preprocess:201808101149--201808101210-v342-dev",
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
			{Name: eosdName, Num: 1}, //  ocr-scene-detect独占 GPU0
		},
		{
			{Name: eoscName, Num: 1}, //  ocr-sari-crann 独占 GPU1
		},
	})
}

func OcrSariIdcardDetectEvalClient(
	client func(method string, respSample interface{}) (endpoint.Endpoint, error),
) middleware.Service {
	end, _ := client("POST", ocridcard.EvalOcrSariIdcardDetectResp{})
	return ocridcard.EvalOcrSariIdcardDetectEndpoints{
		EvalOcrSariIdcardDetectEP: func(ctx context.Context, req0 interface{}) (interface{}, error) {
			req1, _ := req0.(ocridcard.EvalOcrSariIdcardDetectReq)
			var req2 ReqDetect
			req2.Data.URI = string(req1.Data.IMG.URI)
			return end(ctx, req2)
		}}
}

func OcrSariIdcardRecogEvalClient(
	client func(method string, respSample interface{}) (endpoint.Endpoint, error),
) middleware.Service {
	end, _ := client("POST", ocridcard.EvalOcrSariIdcardRecogResp{})
	return ocridcard.EvalOcrSariIdcardRecogEndpoints{
		EvalOcrSariIdcardRecogEP: func(ctx context.Context, req0 interface{}) (interface{}, error) {
			req1, _ := req0.(ocridcard.EvalOcrSariIdcardRecogReq)
			// var req2 ocridcard.EvalOcrSariIdcardRecogReq
			var req2 ReqRecog
			req2.Data.URI = string(req1.Data.IMG.URI)
			req2.Params.Bboxes = req1.Params.Bboxes
			return end(ctx, req2)
		},
	}
}

func OcrSariIdcardPreEvalClient(
	client func(method string, respSample interface{}) (endpoint.Endpoint, error),
) middleware.Service {
	end, _ := client("POST", ocridcard.EvalOcrSariIdcardPreProcessResp{})
	return ocridcard.EvalOcrSariIdcardPreProcessEndpoints{
		EvalOcrSariIdcardPreProcessEP: func(ctx context.Context, req0 interface{}) (interface{}, error) {
			req1, _ := req0.(ocridcard.EvalOcrSariIdcardPreProcessReq)
			var req2 ReqPre
			req2.Data.URI = string(req1.Data.IMG.URI)
			req2.Params = req1.Params
			return end(ctx, req2)
		},
	}
}
