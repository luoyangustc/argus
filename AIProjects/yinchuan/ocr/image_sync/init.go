package image_sync

import (
	"context"

	"github.com/go-kit/kit/endpoint"
	"qiniu.com/argus/AIProjects/yinchuan/ocr"
	"qiniu.com/argus/service/middleware"
	sbiz "qiniu.com/argus/service/scenario/biz"
	scenario "qiniu.com/argus/service/scenario/image_sync"
	"qiniu.com/argus/service/service/biz"
	pimage "qiniu.com/argus/service/service/image"
	"qiniu.com/argus/service/transport"
)

const (
	VERSION = "1.0.0"
)

var (
	_DOC_OCRCLASSIFY = scenario.APIDoc{
		Version: VERSION,
		Desc:    []string{"文本信息识别"},
		Request: `POST /v1/ocr/classify  Http/1.1
Content-Type: application/json
{
	"data": {
		"uri": "http://oqgascup5.bkt.clouddn.com/ocr/WechatIMG61.png"
	}
}`,
		Response: `200 ok
Content-Type: application/json
{
	"code": 0,
	"message": "",
	"result": [
		{
			bboxes: [[134,227],[419,227],[419,262],[134,262]],
       		class: 'idcard_positive',
			index: 1,
			score: 0.9992047
		},{
			bboxes: [[111,222],[333,222],[333,444],[111,444]],
			class: 'bankcard_positive',
			index: 3,
			score: 0.98325944
		},{
			bboxes: [[121,232],[343,232],[343,454],[121,454]],
			class: 'bankcard_positive',
			index: 3,
			score: 0.9056973
		}
	]
}`,
		RequestParam: []scenario.APIDocParam{
			{Name: "uri", Type: "string", Desc: "图片资源地址"},
		},
		ResponseParam: []scenario.APIDocParam{
			{Name: "code", Type: "int", Desc: "0:表示处理成功；不为0:表示出错"},
			{Name: "message", Type: "string", Desc: "描述结果或出错信息"},
			{Name: "bboxes", Type: "list[4,2]", Desc: "返回的图片中所有的文本框位置，为顺时针/逆时针方向旋转的任意四边形[[x1,y1],[x2,y2],[x3,y3],[x4,y4]]注：当是28:blog, 29:wechat, 45:other text, 49:mobile-scereenshot这几类时，其为[[0,0],[0,0],[0,0],[0,0]]"},
			{Name: "class", Type: "string", Desc: "检测物体类型. 1:idcard_positive, 2:idcard_negative, 3:bankcard_positive, 4:bankcard_negative, 5:gongzhang, 28:blog, 29:wechat, 45:other text, 49:mobile-scereenshot"},
			{Name: "index", Type: "int", Desc: "分类号, 对应class中的编号：注：当是28:blog, 29:wechat, 45:other text, 49:mobile-scereenshot这几类时，且阈值小于0.9时，其index为-1"},
			{Name: "score", Type: "float32", Desc: "检测结果的置信度"},
		},
		ErrorMessage: []scenario.APIDocError{},
	}
)

func Import(serviceID string) func(interface{}) {
	return func(is interface{}) {
		Init(is.(scenario.ImageServer), serviceID)
	}
}

func Init(is scenario.ImageServer, serviceID string) {
	var eocSet sbiz.ServiceEvalSetter
	var eoc ocr.EvalOcrClassifyService

	var eotSet sbiz.ServiceEvalSetter
	var eot ocr.EvalOcrTerrorService

	set := is.Register(
		scenario.ServiceInfo{ID: serviceID, Name: "yinchuanclassify", Version: VERSION},
		&middleware.ServiceFactory{
			New: func() middleware.Service {
				{
					eoc1, _ := eocSet.Gen()
					eoc = eoc1.(ocr.EvalOcrClassifyService)
					eot1, _ := eotSet.Gen()
					eot = eot1.(ocr.EvalOcrTerrorService)
				}
				s, _ := ocr.NewOcrClassifyService(eoc, eot)
				return s
			},
			NewShell: func() middleware.ServiceEndpoints { return ocr.OcrClassifyEndpoints{} },
		}).
		UpdateRouter(func(
			sf func() middleware.Service,
			imageParser func() pimage.IImageParse,
			path func(string) *scenario.ServiceRoute,
		) error {
			endp := func() ocr.OcrClassifyEndpoints {
				svc := sf()
				endp, ok := svc.(ocr.OcrClassifyEndpoints)
				if !ok {
					svc, _ = middleware.MakeMiddleware(svc, ocr.OcrClassifyEndpoints{}, nil, nil)
					endp = svc.(ocr.OcrClassifyEndpoints)
				}
				return endp
			}

			type Req struct {
				Data struct {
					URI string `json:"uri"`
				} `json:"data"`
			}

			path("/v1/ocr/classify").Doc(_DOC_OCRCLASSIFY).Route().
				Methods("POST").Handler(transport.MakeHttpServer(
				func(ctx context.Context, req0 interface{}) (interface{}, error) {
					req1, _ := req0.(Req)
					var req2 ocr.OcrClasssifyReq
					var err error
					req2.Data.IMG, err = imageParser().ParseImage(ctx, req1.Data.URI)
					if err != nil {
						return nil, err
					}
					return endp().OcrClassifyEP(ctx, req2)
				},
				Req{}))
			return nil
		})

	eocSet = set.NewEval(
		sbiz.ServiceEvalInfo{Name: "ocr-refinedet", Version: "1.0.0"},
		OcrClassifyEvalClient,
		func() middleware.ServiceEndpoints { return ocr.EvalOcrClassifyEndpoints{} },
	).SetModel(biz.EvalModelConfig{
		//Image: "reg.qiniu.com/avaprd/ava-eval-ataraxia-ocr-route:201810151937--201810151951-v392-dev",
		//Model: "ava-ocr-classify/ocr-refinedet-20181015.tar",
		//Image: "reg.qiniu.com/avaprd/ava-eval-ataraxia-ocr-route:201811211002--201811211050-v449-dev",
		Image: "hub2.qiniu.com/1381102897/ava-eval-ataraxia-ocr-route:201811211002--201811211050-v449-dev",
		Model: "ava-ocr-classify/ocr-refinedet-20181119.tar",
		Args: &biz.ModelConfigArgs{
			BatchSize:    1,
			ImageWidth:   225,
			CustomValues: map[string]interface{}{},
		},
	})

	eotSet = set.NewEval(
		sbiz.ServiceEvalInfo{Name: "ocr-terror", Version: "1.0.0"},
		OcrTerrorEvalClient,
		func() middleware.ServiceEndpoints { return ocr.EvalOcrTerrorEndpoints{} },
	).SetModel(biz.EvalModelConfig{
		Image: "hub2.qiniu.com/1381102897/ava-eval-terror-classify-senet:20180802-v1--201811071101-v424-private",
		Model: "weixinweibo-ocr/text-classification/terror-classify-v0.36-yinchuang_private-20181108.tar",
		Args: &biz.ModelConfigArgs{
			BatchSize: 1,
			CustomValues: map[string]interface{}{
				"threshold": 0.9,
			},
		},
	})
}

type Req struct {
	Data struct {
		URI string `json:"uri"`
	} `json:"data"`
}

func OcrClassifyEvalClient(
	client func(method string, respSample interface{}) (endpoint.Endpoint, error),
) middleware.Service {
	end, _ := client("POST", ocr.EvalOcrRefinedetResp{})
	return ocr.EvalOcrClassifyEndpoints{
		EvalOcrClassifyEP: func(ctx context.Context, req0 interface{}) (interface{}, error) {
			req1, _ := req0.(ocr.EvalOcrRefinedetReq)
			var req2 Req
			req2.Data.URI = string(req1.Data.IMG.URI)
			return end(ctx, req2)
		},
	}
}

func OcrTerrorEvalClient(
	client func(method string, respSample interface{}) (endpoint.Endpoint, error),
) middleware.Service {
	end, _ := client("POST", ocr.EvalOcrTerrorResp{})
	return ocr.EvalOcrTerrorEndpoints{
		EvalOcrTerrorEP: func(ctx context.Context, req0 interface{}) (interface{}, error) {
			req1, _ := req0.(ocr.EvalOcrTerrorReq)
			var req2 Req
			req2.Data.URI = string(req1.Data.IMG.URI)
			return end(ctx, req2)
		},
	}
}
