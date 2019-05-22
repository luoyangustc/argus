package image_sync

import (
	"context"
	"encoding/json"

	"github.com/go-kit/kit/endpoint"
	"github.com/imdario/mergo"
	"qiniu.com/argus/AIProjects/wangan/image/police"
	"qiniu.com/argus/service/middleware"
	"qiniu.com/argus/service/scenario/biz"
	scenario "qiniu.com/argus/service/scenario/image_sync"
	pimage "qiniu.com/argus/service/service/image"
	"qiniu.com/argus/service/transport"
)

const (
	VERSION = "1.0.0"
)

var (
	_DOC_POLICE = scenario.APIDoc{
		Version: VERSION,
		Desc:    []string{`警察图像识别`},
		Request: `POST /v1/police Http/1.1
Content-Type:application/json

{
	"data": {
		"uri": "http://image2.jpeg"
	},
	"params":{
		"detail": true
	}
}`,
		Response: `200 ok
Content-Type:application/json

{
	"code": ,
	"message": "",
	"result": [
		{
			"index": 1,
			"score":  0.996654,
			"class": "police_badge",
			"pts": [[227, 149], [256, 149], [256, 188], [227, 188]]
		},
		...
	]
}`,
		RequestParam: []scenario.APIDocParam{
			{Name: "data.uri", Type: "string", Desc: "图片资源地址"},
			{Name: "params.detail", Type: "bool", Desc: "是否显示详细信息；可选参数,默认为false"},
		},
		ResponseParam: []scenario.APIDocParam{
			{Name: "code", Type: "int", Desc: "0:表示处理成功；不为0:表示出错"},
			{Name: "message", Type: "string", Desc: "描述结果或出错信息"},
			{Name: "result.label", Type: "int", Desc: "标签{0:正常，未识别出所选类别；1:识别涉警物体}"},
			{Name: "result.score", Type: "float", Desc: "识别准确度，取所有识别出的子类别中最高准确度"},
			{Name: "result.detections.[].class", Type: "string", Desc: "图片中识别出的类别,{警徽、警服、警车、警用摩托}"},
			{Name: "result.detections.[].score", Type: "float", Desc: "识别准确度"},
			{Name: "result.detections.[].pts", Type: "list", Desc: "检测识别的物体在图片中的位置，四点坐标值 [左上，右上，右下，左下] 四点坐标框定的物体区域(当detail=true时返回)"},
		},
		ErrorMessage: []scenario.APIDocError{},
	}
)

type Config police.Config

func (c *Config) UnmarshalJSON(b []byte) error {
	type C Config
	c2 := C{}
	if err := json.Unmarshal(b, &c2); err != nil {
		return err
	}
	_ = mergo.Merge(&c2, C(police.DEFAULT))
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
		config = Config(police.DEFAULT)
		eSet   biz.ServiceEvalSetter
	)

	set := is.Register(
		scenario.ServiceInfo{ID: serviceID, Name: "police", Version: VERSION},
		&middleware.ServiceFactory{
			New: func() middleware.Service {
				es, _ := eSet.Gen()
				eval := es.(police.EvalPoliceService)
				s, _ := police.NewPoliceService(police.Config(config), eval)
				return s
			},
			NewShell: func() middleware.ServiceEndpoints { return police.PoliceEndPoints{} },
		},
	).UpdateRouter(func(
		sf func() middleware.Service,
		imageParser func() pimage.IImageParse,
		path func(string) *scenario.ServiceRoute,
	) error {
		endp := func() police.PoliceEndPoints {
			svc := sf()
			endp, ok := svc.(police.PoliceEndPoints)
			if !ok {
				svc, _ := middleware.MakeMiddleware(svc, police.PoliceEndPoints{}, nil, nil)
				endp = svc.(police.PoliceEndPoints)
			}
			return endp
		}

		type PoliceReq struct {
			Data struct {
				URI string `json:"uri"`
			} `json:"data"`
			Params struct {
				Detail bool `json:"detail"`
			} `json:"params"`
		}
		path("/v1/police").Doc(_DOC_POLICE).Route().Methods("POST").Handler((transport.MakeHttpServer(
			func(ctx context.Context, req0 interface{}) (interface{}, error) {
				var (
					req1 police.PoliceReq
					err  error
				)
				req2, _ := req0.(PoliceReq)
				req1.Data.IMG, err = imageParser().ParseImage(ctx, req2.Data.URI)
				req1.Params = req2.Params
				if err != nil {
					return nil, err
				}
				return endp().PoliceEP(ctx, req1)
			},
			PoliceReq{},
		)))

		return nil
	})

	_ = set.GetConfig(context.Background(), &config)

	eSet = set.NewEval(
		biz.ServiceEvalInfo{Name: "evalPolice", Version: "1.0.0"},
		PoliceEvalClient,
		func() middleware.ServiceEndpoints { return police.EvalPoliceEndPoints{} },
	).SetModel(police.EVAL_POLICE_CONFIG).GenId()
}

func PoliceEvalClient(
	client func(method string, respSample interface{}) (endpoint.Endpoint, error),
) middleware.Service {
	end, _ := client("POST", police.EvalPoliceResp{})
	return police.EvalPoliceEndPoints{
		EvalPoliceEP: func(ctx context.Context, req0 interface{}) (interface{}, error) {
			type PoliceReq struct {
				Data struct {
					URI string `json:"uri"`
				} `json:"data"`
			}
			req1, _ := req0.(police.EvalPoliceReq)
			var req2 PoliceReq
			req2.Data.URI = string(req1.Data.IMG.URI)
			return end(ctx, req2)
		}}
}
