package image_sync

import (
	"context"
	"encoding/json"

	"github.com/go-kit/kit/endpoint"
	"github.com/imdario/mergo"
	"qiniu.com/argus/AIProjects/graffiti"
	"qiniu.com/argus/service/middleware"
	"qiniu.com/argus/service/scenario/biz"
	scenario "qiniu.com/argus/service/scenario/image_sync"
	sbiz "qiniu.com/argus/service/service/biz"
	pimage "qiniu.com/argus/service/service/image"
	"qiniu.com/argus/service/transport"
)

const (
	VERSION = "1.0.0"
)

var (
	_DOC_GRAFFITI = scenario.APIDoc{
		Version: VERSION,
		Desc:    []string{`涂鸦检测`},
		Request: `POST /v1/graffiti Http/1.1
Content-Type:application/json

{
	"data": {
		"uri": "http://image2.jpeg"
	}
}`,
		Response: ` 200 ok
Content-Type:application/json

{
	"code": ,
	"message": "",
	"result": {
		"detections": [
			{
				"index":1,
				"class": "graffiti",
				"bounding_box": {
					"pts": [[268,212], [354,212], [354,320], [268,320]],
					"score": 0.9998436
				}
			}
		]
	}
}`,
		RequestParam: []scenario.APIDocParam{
			{Name: "uri", Type: "string", Desc: "图片资源地址"},
		},
		ResponseParam: []scenario.APIDocParam{
			{Name: "code", Type: "int", Desc: "0:表示处理成功；不为0:表示出错"},
			{Name: "message", Type: "string", Desc: "描述结果或出错信息"},
			{Name: "detections", Type: "list", Desc: "检测出的涂鸦列表"},
			{Name: "detections.[].index", Type: "int", Desc: "涂鸦类别序号,当前版本取值范围{1}"},
			{Name: "detections.[].class", Type: "string", Desc: `涂鸦类别名称,当前版本取值范围{"graffiti"}`},
			{Name: "detections.[].bounding_box", Type: "map", Desc: "涂鸦坐标信息"},
			{Name: "detections.[].bounding_box.pts", Type: "list", Desc: "涂鸦在图片中的位置，四点坐标值 [左上，右上，右下，左下] 四点坐标框定的涂鸦区域"},
			{Name: "detections.[].bounding_box.score", Type: "float", Desc: "涂鸦检测的置信度，取值范围0~1，1为准确度最高"},
		},
		ErrorMessage: []scenario.APIDocError{},
	}
)

type Config graffiti.Config

func (c *Config) UnmarshalJSON(b []byte) error {
	type C Config
	c2 := C{}
	if err := json.Unmarshal(b, &c2); err != nil {
		return err
	}
	_ = mergo.Merge(&c2, C(graffiti.DEFAULT))
	*c = Config(c2)
	return nil
}

func Import(serviceID string) func(interface{}) {
	return func(is interface{}) {
		Init(is.(scenario.ImageServer), serviceID)
	}
}

func Init(is scenario.ImageServer, serviceID string) {
	var (
		config = Config(graffiti.DEFAULT)
		eSet   biz.ServiceEvalSetter
	)

	set := is.Register(
		scenario.ServiceInfo{ID: serviceID, Name: "graffiti", Version: VERSION},
		&middleware.ServiceFactory{
			New: func() middleware.Service {
				es, _ := eSet.Gen()
				egd := es.(graffiti.EvalGraffitiDetectService)
				s, _ := graffiti.NewGraffitiService(graffiti.Config(config), egd)
				return s
			},
			NewShell: func() middleware.ServiceEndpoints { return graffiti.GraffitiEndPoints{} },
		},
	).UpdateRouter(func(
		sf func() middleware.Service,
		imageParser func() pimage.IImageParse,
		path func(string) *scenario.ServiceRoute,
	) error {
		endp := func() graffiti.GraffitiEndPoints {
			svc := sf()
			endp, ok := svc.(graffiti.GraffitiEndPoints)
			if !ok {
				svc, _ = middleware.MakeMiddleware(svc, graffiti.GraffitiEndPoints{}, nil, nil)
				endp = svc.(graffiti.GraffitiEndPoints)
			}
			return endp
		}

		type GraffitiReq struct {
			Data struct {
				URI string `json:"uri"`
			} `json:"data"`
		}
		path("/v1/graffiti").Doc(_DOC_GRAFFITI).Route().Methods("POST").Handler(transport.MakeHttpServer(
			func(ctx context.Context, req0 interface{}) (interface{}, error) {
				var (
					req1 graffiti.GraffitiReq
					err  error
				)
				req2, _ := req0.(GraffitiReq)
				req1.Data.IMG, err = imageParser().ParseImage(ctx, req2.Data.URI)
				if err != nil {
					return nil, err
				}
				return endp().GraffitiEP(ctx, req1)
			},
			GraffitiReq{}))
		return nil
	})

	_ = set.GetConfig(context.Background(), &config)

	eSet = set.NewEval(
		biz.ServiceEvalInfo{Name: "graffiti-detect", Version: "1.0.0"},
		GraffitiDetEvalClient,
		func() middleware.ServiceEndpoints { return graffiti.EvalGraffitiDetectEndPoints{} },
	).SetModel(sbiz.EvalModelConfig{
		Image: "hub2.qiniu.com/1381102897/ava-eval-tuya-demo-refinedet:20181204-v2--20181204-v508-private",
		Model: "ava-graffiti-detect/refinenet/20181204-private.tar",
		Args: &sbiz.ModelConfigArgs{
			BatchSize: 5,
		},
		Type: sbiz.EvalRunTypeServing,
	})
}
func GraffitiDetEvalClient(
	client func(method string, respSample interface{}) (endpoint.Endpoint, error),
) middleware.Service {
	end, _ := client("POST", graffiti.GraffitiDetectResp{})
	return graffiti.EvalGraffitiDetectEndPoints{
		EvalGraffitiDetectEP: func(ctx context.Context, req0 interface{}) (interface{}, error) {
			type GraffitiReq struct {
				Data struct {
					URI string `json:"uri"`
				} `json:"data"`
			}
			req1, _ := req0.(graffiti.GraffitiDetectReq)
			var req2 GraffitiReq
			req2.Data.URI = string(req1.Data.IMG.URI)
			return end(ctx, req2)
		}}
}
