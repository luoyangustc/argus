package face

import (
	"context"
	"encoding/json"

	"github.com/go-kit/kit/endpoint"
	"github.com/imdario/mergo"

	"qiniu.com/argus/service/middleware"
	sbiz "qiniu.com/argus/service/scenario/biz"
	scenario "qiniu.com/argus/service/scenario/image_sync"
	"qiniu.com/argus/service/service/biz"
	pimage "qiniu.com/argus/service/service/image"
	"qiniu.com/argus/service/service/image/face"
	"qiniu.com/argus/service/transport"
	"qiniu.com/argus/utility/evals"
)

const (
	VERSION = "1.1.0"
)

var (
	_DOC_SIM = scenario.APIDoc{
		Version: VERSION,
		Desc:    []string{`人脸相似性检测`, `若一张图片中有多个脸则选择最大的脸`},
		Request: `POST /v1/face/sim  Http/1.1
Content-Type:application/json

{
	"data": [
		{
			"uri": "http://image2.jpeg"
		},
		{
			"uri": "http://image1.jpeg
		}
	]
}`,
		Response: ` 200 ok
Content-Type:application/json

{
	"code": 0,
	"message": "success",
	"result": {
		"faces":[{
				"score": 0.987,
				"pts": [[225,195], [351,195], [351,389], [225,389]]
			},
			{
				"score": 0.997,
				"pts": [[225,195], [351,195], [351,389], [225,389]]
			}], 
		"similarity": 0.87,
		"same": 0  
	}	
}`,
		RequestParam: []scenario.APIDocParam{
			{Name: "data", Type: "list", Desc: "两个图片资源地址"},
		},
		ResponseParam: []scenario.APIDocParam{
			{Name: "code", Type: "int", Desc: "0:表示处理成功；不为0:表示出错"},
			{Name: "message", Type: "string", Desc: "描述结果或出错信息"},
			{Name: "faces", Type: "list", Desc: "两张图片中选择出来进行对比的脸"},
			{Name: "score", Type: "float", Desc: "人脸识别的准确度，取值范围0~1，1为准确度最高"},
			{Name: "pts", Type: "list", Desc: "人脸在图片上的坐标"},
			{Name: "similarity", Type: "float", Desc: "人脸相似度，取值范围0~1，1为准确度最高"},
			{Name: "same", Type: "bool", Desc: "是否为同一个人"},
		},
		ErrorMessage: []scenario.APIDocError{
			{Code: 4000601, Desc: "未检测到人脸"},
		},
	}

	_DOC_DETECT = scenario.APIDoc{
		Version: VERSION,
		Desc:    []string{`检测人脸所在位置`, `每次输入一张图片，返回所有检测到的脸的位置`},
		Request: `POST /v1/face/detect  Http/1.1
Content-Type:application/json

{
	"data": {
			"uri": "http://image1.jpeg"
	}
}`,
		Response: ` 200 ok
Content-Type:application/json

{
	"code": 0,
	"message": "",
	"result": {
        "detections": [
            {
                "bounding_box": {
                    "pts": [[268,212], [354,212], [354,320], [268,320]],
                    "score": 0.9998436
                }
            },
            {
                "bounding_box": {
                    "pts": [[159,309], [235,309], [235,408], [159,408]],
                    "score": 0.9997162
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
			{Name: "detections", Type: "list", Desc: "检测出的人脸列表"},
			{Name: "bounding_box", Type: "map", Desc: "人脸坐标信息"},
			{Name: "pts", Type: "list", Desc: "人脸在图片中的位置，四点坐标值 [左上，右上，右下，左下] 四点坐标框定的脸部"},
			{Name: "score", Type: "float", Desc: "人脸的检测置信度，取值范围0~1，1为准确度最高"},
		},
		ErrorMessage: []scenario.APIDocError{},
	}

	_DOC_ATTRIBUTE = scenario.APIDoc{
		Version: VERSION,
		Desc:    []string{`提取人脸属性`, `每次输入一张图片及多个脸，返回所有检测到的人脸属性`},
		Request: `POST /v1/face/attribute  Http/1.1
Content-Type:application/json

{
	"data": {
		"uri": "http://image1.jpeg"
		"attribute": {
			"faces":[
				{
					"orientation": 0.031946659088134769,
					"pts":[
						[23,343],
						[23,434],
						[323,434],
						[323,343]
					],
					"landmarks":[s
						[23,343],
						[23,434],
						[173,388],
						[323,434],
						[323,343]
					]
				},
				...
			]
		}
	},
    "params": {
        "roi_scale": "<roi_scale:float>"
    }
}`,
		Response: ` 200 ok
Content-Type:application/json

{
	"code": 0,
    "message": "error message", // 错误信息
    "result": {
        "faces":[
                {
                    "orientation": 0.031946659088134769,
                    "pts":[
                        [23,343],
                        [23,434],
                        [323,434],
                        [323,343]
                    ],
                    "landmarks":[
                        [23,343],
                        [23,434],
                        [173,388],
                        [323,434],
                        [323,343]
                    ],
					"gender" : {
						"score" : 0.999965786933899,
						"type" : "male"
					},
					"age" : 22.7
                },
                ...
            ]
        }
    }
}`,
		RequestParam: []scenario.APIDocParam{
			{Name: "data.uri", Type: "string", Desc: "图片资源地址"},
			{Name: "data.attribute.faces.[].orientation", Type: "float", Desc: "人脸角度,检测api输出，可选"},
			{Name: "data.attribute.faces.[].pts", Type: "list", Desc: "人脸在图片中的位置，四点坐标值 [左上，右上，右下，左下] 四点坐标框定的脸部"},
			{Name: "data.attribute.faces.[].landmarks", Type: "list", Desc: "人脸五点信息"},
			{Name: "data.params.roi_scale", Type: "float", Desc: "截图缩放比例，默认值：1.0,可选"},
		},
		ResponseParam: []scenario.APIDocParam{
			{Name: "code", Type: "int", Desc: "0:表示处理成功；不为0:表示出错"},
			{Name: "message", Type: "string", Desc: "描述结果或出错信息"},
			{Name: "result.faces.[].orientation", Type: "float", Desc: "人脸角度"},
			{Name: "result.faces.[].pts", Type: "list", Desc: "人脸在图片中的位置，四点坐标值 [左上，右上，右下，左下] 四点坐标框定的脸部"},
			{Name: "result.faces.[].landmarks", Type: "list", Desc: "人脸五点信息"},
			{Name: "result.faces.[].age", Type: "float", Desc: "年龄"},
			{Name: "result.faces.[].gender.type", Type: "string", Desc: "性别，{male,female}"},
			{Name: "result.faces.[].gender.score", Type: "float", Desc: "性别置信度"},
		},
		ErrorMessage: []scenario.APIDocError{},
		Appendix: []string{
			`每张图片每次全部的脸框，pts字段和landmarks字段二选一，如果同时存在，只解析landmarks字段`,
		},
	}
)

type Config face.Config

func (c *Config) UnmarshalJSON(b []byte) error {
	type C Config
	c2 := C{}
	if err := json.Unmarshal(b, &c2); err != nil {
		return err
	}
	_ = mergo.Merge(&c2, face.DEFAULT)
	*c = Config(c2)
	return nil
}

func Import(serviceID string) func(interface{}) {
	return func(is interface{}) {
		Init(is.(scenario.ImageServer), serviceID)
	}
}

func Init(is scenario.ImageServer, serviceID string) {
	var config = Config(face.DEFAULT)

	var eSet1 sbiz.ServiceEvalSetter
	var eSet2 sbiz.ServiceEvalSetter
	var eSet3 sbiz.ServiceEvalSetter

	set := is.Register(
		scenario.ServiceInfo{ID: serviceID, Name: "face", Version: VERSION},
		&middleware.ServiceFactory{
			New: func() middleware.Service {
				ss1, _ := eSet1.Gen()
				ss2, _ := eSet2.Gen()
				ss3, _ := eSet3.Gen()
				s, _ := face.NewFaceService(face.Config(config),
					ss1.(face.EvalFaceDetectService),
					ss2.(face.EvalFaceFeatureService),
					ss3.(face.EvalFaceAttributeService),
				)
				return s
			},
			NewShell: func() middleware.ServiceEndpoints { return face.FaceEndpoints{} },
		}).
		UpdateRouter(func(
			sf func() middleware.Service,
			imageParser func() pimage.IImageParse,
			path func(string) *scenario.ServiceRoute,
		) error {
			endp := func() face.FaceEndpoints {
				svc := sf()
				endp, ok := svc.(face.FaceEndpoints)
				if !ok {
					svc, _ = middleware.MakeMiddleware(svc, face.FaceEndpoints{}, nil, nil)
					endp = svc.(face.FaceEndpoints)
				}
				return endp
			}

			type FaceSimReq struct {
				Data []struct {
					URI string `json:"uri"`
				} `json:"data"`
			}

			type FaceDetectReq struct {
				Data struct {
					URI string `json:"uri"`
				} `json:"data"`
			}

			type FaceAttributeReq struct {
				Data struct {
					URI       string `json:"uri"`
					Attribute struct {
						Faces []struct {
							Orientation float32  `json:"orientation"`
							Pts         [][2]int `json:"pts,omitempty"`
							Landmarks   [][2]int `json:"landmarks,omitempty"`
						} `json:"faces"`
					} `json:"attribute"`
				} `json:"data"`
				Params struct {
					RoiScale float32 `json:"roi_scale"`
				} `json:"params"`
			}

			path("/v1/face/sim").Doc(_DOC_SIM).Route().
				Methods("POST").Handler(transport.MakeHttpServer(
				func(ctx context.Context, req0 interface{}) (interface{}, error) {
					req1, _ := req0.(FaceSimReq)
					var req2 = face.FaceSimReq{Data: []struct{ IMG pimage.Image }{}}
					for _, data := range req1.Data {
						img, err := imageParser().ParseImage(ctx, data.URI)
						if err != nil {
							return nil, err
						}
						req2.Data = append(req2.Data, struct{ IMG pimage.Image }{IMG: img})
					}
					return endp().SimEP(ctx, req2)
				},
				FaceSimReq{}))

			path("/v1/face/detect").Doc(_DOC_DETECT).Route().
				Methods("POST").Handler(transport.MakeHttpServer(
				func(ctx context.Context, req0 interface{}) (interface{}, error) {
					req1, _ := req0.(FaceDetectReq)
					var req2 face.FaceDetectReq
					var err error
					req2.Data.IMG, err = imageParser().ParseImage(ctx, req1.Data.URI)
					if err != nil {
						return nil, err
					}
					return endp().DetectEP(ctx, req2)
				},
				FaceDetectReq{}))

			path("/v1/face/attribute").Doc(_DOC_ATTRIBUTE).Route().
				Methods("POST").Handler(transport.MakeHttpServer(
				func(ctx context.Context, req0 interface{}) (interface{}, error) {
					req1, _ := req0.(FaceAttributeReq)
					var req2 face.FaceAttributeReq
					var err error
					req2.Data.IMG, err = imageParser().ParseImage(ctx, req1.Data.URI)
					if err != nil {
						return nil, err
					}
					req2.Data.Attribute = req1.Data.Attribute
					req2.Params = req1.Params
					return endp().AttributeEP(ctx, req2)
				},
				FaceAttributeReq{}))
			return nil
		})

	_ = set.GetConfig(context.Background(), &config)

	eSet1 = set.NewEval(
		sbiz.ServiceEvalInfo{Name: "facex-detect", Version: "1.0.0"},
		FaceDetEvalClient,
		func() middleware.ServiceEndpoints { return face.EvalFaceDetectEndpoints{} },
	).SetModel(biz.EvalModelConfig{
		Image: "hub2.qiniu.com/1381102897/ava-eval-face.face-det-plus-orient.tron:201809121141",
		Model: "ava-facex-detect/tron-refinenet-mtcnn/20180824-private.tar",
		Args: &biz.ModelConfigArgs{
			BatchSize: 1,
			CustomValues: map[string]interface{}{
				"gpu_id":               0,
				"const_use_quality":    1,
				"output_quality_score": 1,
				"min_face":             50,
			},
		},
		Type: biz.EvalRunTypeServing,
	})

	eSet2 = set.NewEval(
		sbiz.ServiceEvalInfo{Name: "facex-feature-v4", Version: "1.0.0"},
		FaceFeatureEvalClient,
		func() middleware.ServiceEndpoints { return face.EvalFaceFeatureEndpoints{} },
	).SetModel(biz.EvalModelConfig{
		Image: "hub2.qiniu.com/1381102897/ava-eval-face-feature:201808071006",
		Model: "ava-facex-feature-v4/caffe-mxnet/201811191617.tar",
		Args: &biz.ModelConfigArgs{
			BatchSize:  1,
			ImageWidth: 96,
			CustomValues: map[string]interface{}{
				"image_height": 112,
				"input_scale":  0.0078125,
				"workspace":    "/tmp/eval/",
			},
		},
		Type: biz.EvalRunTypeServing,
	})

	eSet3 = set.NewEval(
		sbiz.ServiceEvalInfo{Name: "facex-attribute", Version: "1.0.0"},
		FaceAttributeEvalClient,
		func() middleware.ServiceEndpoints { return face.EvalFaceAttributeEndpoints{} },
	).SetModel(biz.EvalModelConfig{
		Image: "hub2.qiniu.com/1381102897/ava-eval-face-age-gender:pycaffe-cuda8-cudnn7-201903010527--201903028-v901-private",
		Model: "face-attribute/caffe-age-gender/face-age-gender-models.tar",
		Args: &biz.ModelConfigArgs{
			BatchSize: 4,
			CustomValues: map[string]interface{}{
				"max_faces": 8,
				"endian":    "little",
				"model_params": map[string]interface{}{
					"batch_size": 4,
					"cpu_only":   0,
					"gpu_id":     0,
				},
			},
		},
		Type: biz.EvalRunTypeServing,
	})

}

func FaceDetEvalClient(
	client func(method string, respSample interface{}) (endpoint.Endpoint, error),
) middleware.Service {
	end, _ := client("POST", evals.FaceDetectResp{})
	return face.EvalFaceDetectEndpoints{
		EvalFaceDetectEP: func(ctx context.Context, req0 interface{}) (interface{}, error) {
			req1, _ := req0.(face.FaceDetecReq)
			var req2 evals.FaceDetectReq
			req2.Data.URI = string(req1.Data.IMG.URI)
			req2.Params.UseQuality = req1.Params.UseQuality
			return end(ctx, req2)
		}}
}

func FaceFeatureEvalClient(
	client func(method string, respSample interface{}) (endpoint.Endpoint, error),
) middleware.Service {
	end, _ := client("POST", []byte{})
	return face.EvalFaceFeatureEndpoints{
		EvalFaceFeatureEP: func(ctx context.Context, req0 interface{}) (interface{}, error) {
			req1, _ := req0.(face.FaceReq)
			var req2 evals.FaceReq
			req2.Data.URI = string(req1.Data.IMG.URI)
			req2.Data.Attribute = req1.Data.Attribute
			return end(ctx, req2)
		}}
}

func FaceAttributeEvalClient(
	client func(method string, respSample interface{}) (endpoint.Endpoint, error),
) middleware.Service {
	end, _ := client("POST", face.FaceAttributeResp{})
	return face.EvalFaceAttributeEndpoints{
		EvalFaceAttributeEP: func(ctx context.Context, req0 interface{}) (interface{}, error) {
			type FaceAttributeReq struct {
				Data struct {
					URI       string `json:"uri"`
					Attribute struct {
						Faces []struct {
							Orientation float32  `json:"orientation"`
							Pts         [][2]int `json:"pts,omitempty"`
							Landmarks   [][2]int `json:"landmarks,omitempty"`
						} `json:"faces"`
					} `json:"attribute"`
				} `json:"data"`
				Params struct {
					RoiScale float32 `json:"roi_scale"`
				} `json:"params"`
			}
			req1, _ := req0.(face.FaceAttributeReq)
			var req2 FaceAttributeReq
			req2.Data.URI = string(req1.Data.IMG.URI)
			req2.Data.Attribute = req1.Data.Attribute
			req2.Params = req1.Params
			return end(ctx, req2)
		}}
}
