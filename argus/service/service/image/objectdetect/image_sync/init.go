package image_sync

import (
	"context"

	"github.com/go-kit/kit/endpoint"
	"qiniu.com/argus/service/middleware"
	sbiz "qiniu.com/argus/service/scenario/biz"
	scenario "qiniu.com/argus/service/scenario/image_sync"
	"qiniu.com/argus/service/service/biz"
	pimage "qiniu.com/argus/service/service/image"
	"qiniu.com/argus/service/service/image/objectdetect"
	"qiniu.com/argus/service/transport"
)

const (
	VERSION = "1.0.0"
)

// eval 统一名称
const (
	eodName = "object-detect"
)

var (
	_DOC_OBJECTDETECT = scenario.APIDoc{
		Version: VERSION,
		Desc:    []string{"物品检测"},
		Request: `POST /v1/detect  Http/1.1
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
        "detections":[
            {
                "index":1,
                "class":"Person",
                "score":0.28927702,
                "pts":[[ 47,92 ],[ 573,92 ],[ 573,576 ],[ 47,576 ] ],
                "label_cn":"人"
            },
            {
                "index":2,
                "class":"Clothing",
                "score":0.8579982,
                "pts":[[ 46,274 ],[ 572,274 ],[ 572,585 ],[ 46,585 ] ],
                "label_cn":"服饰"
            },
            {
                "index":2,
                "class":"Clothing",
                "score":0.103042044,
                "pts":[[ 0,299 ],[ 199,299 ],[ 199,595 ],[ 0,595 ] ],
                "label_cn":"服饰"
            },
            {
                "index":4,
                "class":"Face",
                "score":0.89105374,
                "pts":[[ 223,109 ],[ 540,109 ],[ 540,397 ],[ 223,397 ] ],
                "label_cn":"人脸，面部"
            },
            {
                "index":7,
                "class":"Woman",
                "score":0.367797,
                "pts":[[ 24,69 ],[ 599,69 ],[ 599,569 ],[ 24,569 ] ],
                "label_cn":"女人"
            },
            {
                "index":12,
                "class":"Girl",
                "score":0.37917978,
                "pts":[[ 25,61 ],[ 578,61 ],[ 578,596 ],[ 25,596 ] ],
                "label_cn":"女孩"
            },
            {
                "index":39,
                "class":"Head",
                "score":0.10832175,
                "pts":[[ 214,73 ],[ 536,73 ],[ 536,383 ],[ 214,383 ] ],
                "label_cn":"头部"
            },
            {
                "index":55,
                "class":"Nose",
                "score":0.44342348,
                "pts":[[ 299,223 ],[ 377,223 ],[ 377,316 ],[ 299,316 ] ],
                "label_cn":"鼻子"
            },
            {
                "index":62,
                "class":"Eye",
                "score":0.24424575,
                "pts":[[ 298,168 ],[ 372,168 ],[ 372,236 ],[ 298,236 ] ],
                "label_cn":"眼睛"
            },
            {
                "index":62,
                "class":"Eye",
                "score":0.20755252,
                "pts":[[ 396,230 ],[ 468,230 ],[ 468,305 ],[ 396,305 ] ],
                "label_cn":"眼睛"
            },
            {
                "index":62,
                "class":"Eye",
                "score":0.17607109,
                "pts":[[ 313,172 ],[ 357,172 ],[ 357,205 ],[ 313,205 ] ],
                "label_cn":"眼睛"
            },
            {
                "index":64,
                "class":"Mouth",
                "score":0.40049475,
                "pts":[[ 266,294 ],[ 339,294 ],[ 339,357 ],[ 266,357 ] ],
                "label_cn":"嘴巴"
            },
            {
                "index":268,
                "class":"Lipstick",
                "score":0.27580106,
                "pts":[[ 270,290 ],[ 338,290 ],[ 338,349 ],[ 270,349 ] ],
                "label_cn":"口红"
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
			{Name: "index", Type: "int", Desc: "物品编号"},
			{Name: "class", Type: "string", Desc: "物品英文名称"},
			{Name: "score", Type: "float32", Desc: "置信度"},
			{Name: "pts", Type: "list", Desc: "返回该物品对应的位置，为顺时针/逆时针方向旋转的任意四边形[[x1,y1],[x2,y2],[x3,y3],[x4,y4]]"},
			{Name: "label_cn", Type: "string", Desc: "物品中文名"},
		},
		ErrorMessage: []scenario.APIDocError{
			{Code: 4000801, Desc: "未检测到物品信息"},
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
	var eod objectdetect.EvalObjectDetectService

	set := is.Register(
		scenario.ServiceInfo{ID: serviceID, Name: "objectdetect", Version: VERSION},
		&middleware.ServiceFactory{
			New: func() middleware.Service {
				{
					eod1, _ := eodSet.Gen()
					eod = eod1.(objectdetect.EvalObjectDetectService)
				}
				s, _ := objectdetect.NewObjectDetectService(eod)
				return s
			},
			NewShell: func() middleware.ServiceEndpoints { return objectdetect.ObjectDetectEndpoints{} },
		}).
		UpdateRouter(func(
			sf func() middleware.Service,
			imageParser func() pimage.IImageParse,
			path func(string) *scenario.ServiceRoute,
		) error {
			endp := func() objectdetect.ObjectDetectEndpoints {
				svc := sf()
				endp, ok := svc.(objectdetect.ObjectDetectEndpoints)
				if !ok {
					svc, _ = middleware.MakeMiddleware(svc, objectdetect.ObjectDetectEndpoints{}, nil, nil)
					endp = svc.(objectdetect.ObjectDetectEndpoints)
				}
				return endp
			}

			type Req struct {
				Data struct {
					URI string `json:"uri"`
				} `json:"data"`
			}

			path("/v1/detect").Doc(_DOC_OBJECTDETECT).Route().
				Methods("POST").Handler(transport.MakeHttpServer(
				func(ctx context.Context, req0 interface{}) (interface{}, error) {
					req1, _ := req0.(Req)
					var req2 objectdetect.DetectionReq
					var err error
					req2.Data.IMG, err = imageParser().ParseImage(ctx, req1.Data.URI)
					if err != nil {
						return nil, err
					}
					return endp().ObjectDetectEP(ctx, req2)
				},
				Req{}))
			return nil
		})

	eodSet = set.NewEval(
		sbiz.ServiceEvalInfo{Name: eodName, Version: "1.0.0"},
		ObjectDetectEvalClient,
		func() middleware.ServiceEndpoints { return objectdetect.EvalObjectDetectEndpoints{} },
	).SetModel(biz.EvalModelConfig{
		Image: "hub2.qiniu.com/1381102897/ava-detect-545:20181205-v3",
		Model: "ava-detection/caffe-detect/caffe-det-545-20171225.tar",
		CustomFiles: map[string]string{
			"label_cn.csv": "ava-detection/label_cn/2018011118.csv",
		},
		Args: &biz.ModelConfigArgs{
			BatchSize:  1,
			ImageWidth: 225,
			CustomValues: map[string]interface{}{
				"threshold": 0.1,
				"other_labels": map[string]string{
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

func ObjectDetectEvalClient(
	client func(method string, respSample interface{}) (endpoint.Endpoint, error),
) middleware.Service {
	end, _ := client("POST", objectdetect.EvalObjectDetectResp{})
	return objectdetect.EvalObjectDetectEndpoints{
		EvalObjectDetectEP: func(ctx context.Context, req0 interface{}) (interface{}, error) {
			req1, _ := req0.(objectdetect.EvalObjectDetectReq)
			var req2 Req
			req2.Data.URI = string(req1.Data.IMG.URI)
			return end(ctx, req2)
		},
	}
}
