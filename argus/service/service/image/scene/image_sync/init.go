package image_sync

import (
	"context"

	"github.com/go-kit/kit/endpoint"
	"qiniu.com/argus/service/middleware"
	sbiz "qiniu.com/argus/service/scenario/biz"
	scenario "qiniu.com/argus/service/scenario/image_sync"
	"qiniu.com/argus/service/service/biz"
	pimage "qiniu.com/argus/service/service/image"
	"qiniu.com/argus/service/service/image/scene"
	"qiniu.com/argus/service/transport"
)

const (
	VERSION = "1.0.0"
)

// eval 统一名称
const (
	eodName = "scene"
)

var (
	_DOC_SCENE = scenario.APIDoc{
		Version: VERSION,
		Desc:    []string{"场景检测"},
		Request: `POST /v1/scene  Http/1.1
Content-Type:application/json
{
	"data":{
		"uri": "http://p9abk98u4.bkt.clouddn.com/103b8c4b581a672d2e560a9f401688865a1b45db64f3.crop.png.jpg"
	}
}`,
		Response: `200 ok
Content-Type:application/json
{
	"code":0,
	"message":"",
	"result":{
		"confidences":[{
			"class":"kindergarden_classroom",
			"index":203,
			"label":["indoor","public"],
			"score":0.9925695,
			"label_cn":"幼儿园教室"
		}]
	}
}`,
		RequestParam: []scenario.APIDocParam{
			{Name: "uri", Type: "string", Desc: "图片资源地址"},
		},
		ResponseParam: []scenario.APIDocParam{
			{Name: "code", Type: "int", Desc: "0:表示处理成功；不为0:表示出错"},
			{Name: "message", Type: "string", Desc: "描述结果或出错信息"},
			{Name: "label", Type: "string", Desc: "图片场景标签"},
			{Name: "score", Type: "float32", Desc: "置信度"},
			{Name: "label_cn", Type: "string", Desc: "中文标签"},
		},
		ErrorMessage: []scenario.APIDocError{
			{Code: 4000801, Desc: "未检测到场景信息"},
		},
	}
)

func Import(serviceID string) func(interface{}) {
	return func(is interface{}) {
		Init(is.(scenario.ImageServer), serviceID)
	}
}

func Init(is scenario.ImageServer, serviceID string) {
	var eodSet sbiz.ServiceEvalSetter
	var eod scene.EvalSceneService

	set := is.Register(
		scenario.ServiceInfo{ID: serviceID, Name: "scene", Version: VERSION},
		&middleware.ServiceFactory{
			New: func() middleware.Service {
				{
					eod1, _ := eodSet.Gen()
					eod = eod1.(scene.EvalSceneService)
				}
				s, _ := scene.NewSceneService(eod)
				return s
			},
			NewShell: func() middleware.ServiceEndpoints { return scene.SceneEndpoints{} },
		}).
		UpdateRouter(func(
			sf func() middleware.Service,
			imageParser func() pimage.IImageParse,
			path func(string) *scenario.ServiceRoute,
		) error {
			endp := func() scene.SceneEndpoints {
				svc := sf()
				endp, ok := svc.(scene.SceneEndpoints)
				if !ok {
					svc, _ = middleware.MakeMiddleware(svc, scene.SceneEndpoints{}, nil, nil)
					endp = svc.(scene.SceneEndpoints)
				}
				return endp
			}

			type Req struct {
				Data struct {
					URI string `json:"uri"`
				} `json:"data"`
			}

			path("/v1/scene").Doc(_DOC_SCENE).Route().
				Methods("POST").Handler(transport.MakeHttpServer(
				func(ctx context.Context, req0 interface{}) (interface{}, error) {
					req1, _ := req0.(Req)
					var req2 scene.SceneReq
					var err error
					req2.Data.IMG, err = imageParser().ParseImage(ctx, req1.Data.URI)
					if err != nil {
						return nil, err
					}
					return endp().SceneEP(ctx, req2)
				},
				Req{}))
			return nil
		})

	eodSet = set.NewEval(
		sbiz.ServiceEvalInfo{Name: eodName, Version: "1.0.0"},
		SceneEvalClient,
		func() middleware.ServiceEndpoints { return scene.EvalSceneEndpoints{} },
	).SetModel(biz.EvalModelConfig{
		Image: "hub2.qiniu.com/1381102897/ava-eval-caffe-classify:20180111-v610-serving-v2",
		Model: "ava-scene/caffe-classify/201712251924.tar",
		CustomFiles: map[string]string{
			"taglist_file": "ava-scene/caffe-classify/201712261635.taglist_file",
			"label_cn.csv": "ava-scene/label_cn/2018011118.csv",
		},
		Args: &biz.ModelConfigArgs{
			BatchSize:  1,
			ImageWidth: 225,
			CustomValues: map[string]interface{}{
				"threshold":  0.1,
				"mean_value": []float32{105.448, 113.768, 116.052},
				"other_labels": map[string]interface{}{
					"cn": "label_cn.csv",
				},
			},
		},
		Type: biz.EvalRunTypeServing,
	})

	// 添加一种1卡的部署方式
	set.AddEvalsDeployModeOnGPU("", [][]sbiz.ServiceEvalDeployProcess{
		{
			{Name: eodName, Num: 1}, // ctpn 独占 GPU0
		},
	})
}

type Req struct {
	Data struct {
		URI string `json:"uri"`
	} `json:"data"`
}

func SceneEvalClient(
	client func(method string, respSample interface{}) (endpoint.Endpoint, error),
) middleware.Service {
	end, _ := client("POST", scene.EvalSceneResp{})
	return scene.EvalSceneEndpoints{
		EvalSceneEP: func(ctx context.Context, req0 interface{}) (interface{}, error) {
			req1, _ := req0.(scene.EvalSceneReq)
			var req2 Req
			req2.Data.URI = string(req1.Data.IMG.URI)
			return end(ctx, req2)
		},
	}
}
