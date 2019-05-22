package image_sync

import (
	"context"
	"encoding/json"

	"github.com/imdario/mergo"

	"qiniu.com/argus/service/middleware"
	sbiz "qiniu.com/argus/service/scenario/biz"
	scenario "qiniu.com/argus/service/scenario/image_sync"
	pimage "qiniu.com/argus/service/service/image"
	"qiniu.com/argus/service/service/image/ads"
	adses "qiniu.com/argus/service/service/image/ads/image_sync"
	"qiniu.com/argus/service/service/image/censor"
	"qiniu.com/argus/service/service/image/politician"
	politicians "qiniu.com/argus/service/service/image/politician/image_sync"
	"qiniu.com/argus/service/service/image/pulp"
	pulps "qiniu.com/argus/service/service/image/pulp/image_sync"
	"qiniu.com/argus/service/service/image/terror"
	terrors "qiniu.com/argus/service/service/image/terror/image_sync"
	"qiniu.com/argus/service/transport"
)

const (
	VERSION = "1.2.0"
)

const (
	PulpFlag       = 1 << iota // 1
	TerrorFlag                 // 2
	PoliticianFlag             // 4
	AdsFlag                    // 8

	PulpTerrorPoliticianFlag = PulpFlag | TerrorFlag | PoliticianFlag           // 7
	AllFlag                  = PulpFlag | TerrorFlag | PoliticianFlag | AdsFlag // 15
)

type Config censor.Config

func (c *Config) UnmarshalJSON(b []byte) error {
	type C Config
	c2 := C{}
	if err := json.Unmarshal(b, &c2); err != nil {
		return err
	}
	c1 := C(censor.DEFAULT)
	_ = mergo.Merge(&c2, &c1)
	*c = Config(c2)
	return nil
}

var (
	_DOC_CENSOR = scenario.APIDoc{
		Version: VERSION,
		Desc:    []string{`图片审核`},
		Request: `POST /v1/image/censor Http/1.1
Content-Type: application/json

{
	"data": {
		"uri": "http://xxx/xxx.jpg"
	},
	"params": {
		"type": [
			"pulp",
			"terror",
			"politician"
		],
		"detail": true
	}
}`,
		Response: ` 200 ok
Content-Type:application/json

{
	"code": 0,
	"message": "",
	"result": {
		"label": 1,
		"score": 0.888,
		"review": true,
		"details": [
			{
				"type": "pulp",
				"label": 1,
				"score": 0.8726697,
				"review": false
			},
			{
				"type": "terror",
				"label": 1,
				"class": <class>,
				"score": 0.6530496,
				"review": false
			},
			{
				"type": "politician",
				"label": 1,
				"score": 0.77954,
				"review": true,
				"more": [
					{
						"boundingBox":{
							"pts": [[1213,400],[205,400],[205,535],[1213,535]],
							"score":0.998
						},
						"value": {
							"name": "xx",
							"score":0.567,
							"review": true
						},
						"sample": {
							"url": "",
							"pts": [[1213,400],[205,400],[205,535],[1213,535]]
						}
					},
					{
						"boundingBox":{
							"pts": [[1109,500],[205,500],[205,535],[1109,535]],
							"score":0.98
						},
						"value": {
							"score":0.987,
							"review": false
						}
					}
				]
			}
		]
	}
}`,
		RequestParam: []scenario.APIDocParam{
			{Name: "uri", Type: "string", Desc: "图片资源地址"},
			{Name: "params.type", Type: "string", Desc: "选择的审核类型，可选项：'pulp'/'terror'/'politician'；可选参数，不填表示全部执行"},
			{Name: "params.detail", Type: "bool", Desc: "是否显示详细信息；可选参数"}},
		ResponseParam: []scenario.APIDocParam{
			{Name: "code", Type: "int", Desc: "0:表示正确"},
			{Name: "message", Type: "string", Desc: "结果描述信息"},
			{Name: "result.label", Type: "int", Desc: "是否违规，0：不违规；1：违规"},
			{Name: "result.score", Type: "float", Desc: "是否违规置信度"},
			{Name: "result.review", Type: "bool", Desc: "整体审核结果是否需要人工复审该图片。true需要false不需要"},
			{Name: "result.details.type", Type: "string", Desc: "审核类型，与用户设置的params.type一致，结果中包含该审核类型的返回结果"},
			{Name: "result.details.label", Type: "int", Desc: "审核结果类别，具体看各类型"},
			{Name: "result.details.class", Type: "string", Desc: "暴恐的详细分类,请参考图片鉴暴恐文档"},
			{Name: "result.details.more", Type: "list", Desc: "当审核类型是politician时，返回人脸的具体信息，请参考图片敏感人物识别文档"},
			{Name: "result.details.score", Type: "float", Desc: "审核结果置信度"},
			{Name: "result.details.review", Type: "bool", Desc: "针对每项审核类型的结果是否需要人工复审该图片。true需要false不需要"},
		},
		ErrorMessage: []scenario.APIDocError{},
	}
)

var (
	_DOC_CENSOR_PREMIER = scenario.APIDoc{
		Version: VERSION,
		Desc:    []string{`图片审核`},
		Request: `POST /v3/censor/image Http/1.1
Content-Type: application/json

{
	"data": {
		"uri":"http://xxx/xxx.jpg"
	},
	"params": {
		"scenes": [
			"ads",
			"pulp",
			"terror",
			"politician"
		]
	}
}`,
		Response: ` 200 ok
Content-Type:application/json

{
	"code": 200,
	"message": "OK",
	"result": {
		"suggestion": "block",
		"scenes": {
			"ads": {
				"suggestion": "block",
				"details": [
					{
						"suggestion": "block",
						"label": "qr_code",
						"score": 0.99999046,
						"detections": [
							{
								"pts": [[176,529],[891,529],[891,1269],[176,1269]],
								"score": 0.99999046
							}
						]
					}
				]
			},
			"politician": {
				"suggestion": "review",
				"details": [
					{
						"suggestion": "review",
						"label": "xxx",
						"group": "domestic_statesman",
						"score": 0.9999962,
						"detections": [
							{
								"pts": [[362,110],[489,110],[489,289],[362,289]],
								"score": 0.9999962
							}
						]
					}
				]
			},
			"pulp": {
				"suggestion": "pass",
				"details": [
					{
						"suggestion": "pass",
						"label": "normal",
						"score": 0.9999974
					}
				]
			},
			"terror": {
				"suggestion": "block",
				"details": [
					{
						"suggestion": "review",
						"label": "bloodiness",
						"score": 0.63997413
					},
					{
						"suggestion": "block",
						"label": "guns",
						"score": 0.98245511,
						"detections": [
							{
								"pts": [[107,1142],[781,1142],[781,1657],[107,1657]],
								"score": 0.98245511
							},
							{
								"pts": [[107,1142],[781,1142],[781,1657],[107,1657]],
								"score": 0.88984421
							}
						]
					}
				]
			}
		}
	}
}`,
		RequestParam: []scenario.APIDocParam{
			{Name: "data.uri", Type: "map", Desc: "多个图片资源地址和id"},
			{Name: "params.scenes", Type: "list", Desc: `审核场景，必选。目前支持"pulp", "terror", "politician", "ads"`},
		},
		ResponseParam: []scenario.APIDocParam{
			{Name: "code", Type: "int", Desc: "处理状态，200表示调用成功"},
			{Name: "message", Type: "string", Desc: "与code对应的状态描述信息"},
			{Name: "result.suggestion", Type: "string", Desc: "图片的综合审核结果：pass/block/review"},
			{Name: "result.scenes.<scene>", Type: "map", Desc: "用户设置的scenes场景的返回结果"},
			{Name: "result.scenes.<scene>.suggestion", Type: "string", Desc: "场景的审核结果：pass/block/review"},
			{Name: "result.scenes.<scene>.details", Type: "list", Desc: "场景的详细信息列表"},
			{Name: "result.scenes.<scene>.details.suggestion", Type: "string", Desc: "详细信息的审核结果：pass/block/review"},
			{Name: "label", Type: "string", Desc: "详细信息的标签，不同场景下有不同的返回值，见附录"},
			{Name: "group", Type: "string", Desc: "politician场景下政治人物的分类，见附录"},
			{Name: "score", Type: "float", Desc: "该审核结果的置信度"},
			{Name: "detections", Type: "list", Desc: "该标签在图片中的检测框列表"},
			{Name: "detections.[].pts", Type: "list", Desc: "坐标框[左上，右上，右下，左下]四个点的横纵坐标值（以图片左上角为原点）"},
			{Name: "detections.[].score", Type: "float", Desc: "该检测的置信度"},
			{Name: "detections.[].comments", Type: "float", Desc: `广告审核，当审核结果label为"ads"时，该字段用于返回识别到的敏感词`},
		},
		ErrorMessage: []scenario.APIDocError{},
		Appendix: []string{`鉴黄标签取值及说明：

| 标签 | 说明 |
| :--- | :--- |
| pulp | 色情 |
| sexy | 性感 |
| normal | 正常 |
`, `暴恐标签取值及说明：

| 标签 | 说明 |
| :--- | :--- |
| illegal_flag | 违规旗帜 |
| knives | 刀 |
| guns | 枪 |
| anime_knives | 二次元刀 |
| anime_guns | 二次元枪 |
| bloodiness | 血腥 |
| bomb | 爆炸 |
| self_burning | 自焚 |
| beheaded | 行刑斩首 |
| march_crowed | 非法集会 |
| fight_police | 警民冲突 |
| fight_person | 打架斗殴 |
| army | 军队 |
| special_characters | 特殊字符 |
| anime_bloodiness | 二次元血腥 |
| special_clothing | 特殊着装 |
| bloodiness_animal | 动物血腥 |
| fire_weapon |  武器发射火焰 |
| normal | 正常 |
`, `政治人物分组标签取值及说明：

| 标签 | 说明 |
| :--- | :--- |
| domestic_statesman | 国内政治人物 |
| foreign_statesman | 国外政治人物 |
| affairs_official_gov | 落马官员（政府) |
| affairs_official_ent | 落马官员（企事业）|
| anti_china_people | 反华分子 |
| terrorist | 恐怖分子 |
| affairs_celebrity | 劣迹艺人 |
| chinese_martyr | 烈士 |
`, `广告标签取值及说明：

| 标签 | 说明 |
| :--- | :--- |
| qr_code | 二维码 |
| bar_code | 条形码 |
| ads | 广告详情，该标签下会返回检测到的广告坐标框和其中的敏感词 |
| summary_ads | 整体广告总结 |
`},
	}
)

func Import(serviceID string) func(interface{}) {
	return func(is interface{}) {
		Init(is.(scenario.ImageServer), serviceID)
	}
}

func Init(is scenario.ImageServer, serviceID string) {

	var config = Config(censor.DEFAULT)

	var (
		epps pulp.EvalPulpService
		epfs pulp.EvalPulpFilterService
		efds politician.EvalFaceDetectService
		effs politician.EvalFaceFeatureService
		epts politician.EvalPoliticianService
		etms terror.EvalTerrorMixupService
		etds terror.EvalTerrorDetectService
		eaqs ads.EvalAdsQrcodeService
		eads ads.EvalAdsDetectService
		ears ads.EvalAdsRecognitionService
		eacs ads.EvalAdsClassifierService
	)

	var (
		eppSet sbiz.ServiceEvalSetter
		epfSet sbiz.ServiceEvalSetter
		efdSet sbiz.ServiceEvalSetter
		effSet sbiz.ServiceEvalSetter
		eptSet sbiz.ServiceEvalSetter
		etmSet sbiz.ServiceEvalSetter
		etdSet sbiz.ServiceEvalSetter
		eaqSet sbiz.ServiceEvalSetter
		eadSet sbiz.ServiceEvalSetter
		earSet sbiz.ServiceEvalSetter
		eacSet sbiz.ServiceEvalSetter
	)

	set := is.Register(
		scenario.ServiceInfo{ID: serviceID, Name: "censor", Version: VERSION},
		&middleware.ServiceFactory{
			New: func() middleware.Service {
				{
					if adses.ON {
						ss, _ := eaqSet.Gen()
						eaqs = ss.(ads.EvalAdsQrcodeService)
						ss, _ = eadSet.Gen()
						eads = ss.(ads.EvalAdsDetectService)
						ss, _ = earSet.Gen()
						ears = ss.(ads.EvalAdsRecognitionService)
						ss, _ = eacSet.Gen()
						eacs = ss.(ads.EvalAdsClassifierService)
					}
					if pulps.ON {
						ss, _ := eppSet.Gen()
						epps = ss.(pulp.EvalPulpService)
						ss, _ = epfSet.Gen()
						epfs = ss.(pulp.EvalPulpFilterService)
					}
					if politicians.ON {
						ss, _ := efdSet.Gen()
						efds = ss.(politician.EvalFaceDetectService)
						ss, _ = effSet.Gen()
						effs = ss.(politician.EvalFaceFeatureService)
						ss, _ = eptSet.Gen()
						epts = ss.(politician.EvalPoliticianService)
					}
					if terrors.ON {
						ss, _ := etmSet.Gen()
						etms = ss.(terror.EvalTerrorMixupService)
						ss, _ = etdSet.Gen()
						etds = ss.(terror.EvalTerrorDetectService)
					}
				}

				s, _ := censor.NewCensorService(censor.Config(config), epps, epfs, efds, effs, epts, etms, etds, eaqs, eads, ears, eacs)
				return s
			},
			NewShell: func() middleware.ServiceEndpoints { return censor.CensorEndpoints{} },
		}).
		UpdateRouter(func(
			sf func() middleware.Service,
			imageParser func() pimage.IImageParse,
			path func(string) *scenario.ServiceRoute,
		) error {
			endp := func() censor.CensorEndpoints {
				svc := sf()
				endp, ok := svc.(censor.CensorEndpoints)
				if !ok {
					svc, _ = middleware.MakeMiddleware(svc, censor.CensorEndpoints{}, nil, nil)
					endp = svc.(censor.CensorEndpoints)
				}
				return endp
			}

			type ImageCensorReq struct {
				Data struct {
					URI string `json:"uri"`
				} `json:"data"`
				Params struct {
					Type   []string `json:"type,omitempty"`
					Detail bool     `json:"detail"`
				} `json:"params,omitempty"`
			}

			path("/v1/image/censor").Doc(_DOC_CENSOR).Route().
				Methods("POST").Handler(transport.MakeHttpServer(
				func(ctx context.Context, req0 interface{}) (interface{}, error) {
					req1, _ := req0.(ImageCensorReq)
					var req2 censor.ImageCensorReq
					var err error
					req2.Params.Type = req1.Params.Type
					req2.Params.Detail = req1.Params.Detail
					req2.Data.IMG, err = imageParser().ParseImage(ctx, req1.Data.URI)
					if err != nil {
						return nil, err
					}
					return endp().CensorEP(ctx, req2)
				},
				ImageCensorReq{}))

			path("/v3/censor/image").Doc(_DOC_CENSOR_PREMIER).Route().
				Methods("POST").Handler(transport.MakeHttpServer(
				func(ctx context.Context, req0 interface{}) (interface{}, error) {
					req1, _ := req0.(censor.IPremierCensorRequest)
					var err error
					req1.Data.IMG, err = imageParser().ParseImage(ctx, req1.Data.URI)
					if err != nil {
						return nil, err
					}
					return endp().PremierCensorEP(ctx, req1)
				},
				censor.IPremierCensorRequest{}))
			return nil
		})

	{
		_ = set.GetConfig(context.Background(), &config)
	}

	var censorFlag int

	if pulps.ON {
		censorFlag |= PulpFlag
		eppSet = set.NewEvalDirect(
			sbiz.ServiceEvalInfo{Name: "censor-evalPulp", Version: "1.1.0"},
			&middleware.ServiceFactory{
				New:      func() middleware.Service { return pulps.ESet.Kernel() },
				NewShell: func() middleware.ServiceEndpoints { return pulp.EvalPulpEndpoints{} },
			},
		)

		epfSet = set.NewEvalDirect(
			sbiz.ServiceEvalInfo{Name: "censor-evalPulpFilter", Version: "1.1.0"},
			&middleware.ServiceFactory{
				New:      func() middleware.Service { return pulps.ESSet.Kernel() },
				NewShell: func() middleware.ServiceEndpoints { return pulp.EvalPulpFilterEndpoints{} },
			},
		)
	}
	if politicians.ON {
		censorFlag |= PoliticianFlag
		efdSet = set.NewEvalDirect(
			sbiz.ServiceEvalInfo{Name: "censor-evalFaceDetect", Version: "1.0.0"},
			&middleware.ServiceFactory{
				New:      func() middleware.Service { return politicians.ESet1.Kernel() },
				NewShell: func() middleware.ServiceEndpoints { return politician.EvalFaceDetectEndpoints{} },
			},
		)

		effSet = set.NewEvalDirect(
			sbiz.ServiceEvalInfo{Name: "censor-evalFacexFeatureV4", Version: "1.1.0"},
			&middleware.ServiceFactory{
				New:      func() middleware.Service { return politicians.ESet2.Kernel() },
				NewShell: func() middleware.ServiceEndpoints { return politician.EvalFaceFeatureEndpoints{} },
			},
		)

		eptSet = set.NewEvalDirect(
			sbiz.ServiceEvalInfo{Name: "censor-evalPolitician", Version: "1.1.0"},
			&middleware.ServiceFactory{
				New:      func() middleware.Service { return politicians.ESet3.Kernel() },
				NewShell: func() middleware.ServiceEndpoints { return politician.EvalPoliticianEndpoints{} },
			},
		)
	}
	if terrors.ON {
		censorFlag |= TerrorFlag
		etmSet = set.NewEvalDirect(
			sbiz.ServiceEvalInfo{Name: "censor-evalTerrorMixup", Version: "1.1.0"},
			&middleware.ServiceFactory{
				New:      func() middleware.Service { return terrors.ESet1.Kernel() },
				NewShell: func() middleware.ServiceEndpoints { return terror.EvalTerrorMixupEndpoints{} },
			},
		)

		etdSet = set.NewEvalDirect(
			sbiz.ServiceEvalInfo{Name: "censor-evalTerrorDetect", Version: "1.1.0"},
			&middleware.ServiceFactory{
				New:      func() middleware.Service { return terrors.ESet2.Kernel() },
				NewShell: func() middleware.ServiceEndpoints { return terror.EvalTerrorDetectEndpoints{} },
			},
		)
	}

	if adses.ON {
		censorFlag |= AdsFlag
		eaqSet = set.NewEvalDirect(
			sbiz.ServiceEvalInfo{Name: "censor-evalAdsQrcode", Version: "1.0.0"},
			&middleware.ServiceFactory{
				New:      func() middleware.Service { return adses.ESet1.Kernel() },
				NewShell: func() middleware.ServiceEndpoints { return ads.EvalAdsQrcodeEndpoints{} },
			},
		)

		eadSet = set.NewEvalDirect(
			sbiz.ServiceEvalInfo{Name: "censor-evalAdsDetect", Version: "1.0.0"},
			&middleware.ServiceFactory{
				New:      func() middleware.Service { return adses.ESet2.Kernel() },
				NewShell: func() middleware.ServiceEndpoints { return ads.EvalAdsDetectEndpoints{} },
			},
		)

		earSet = set.NewEvalDirect(
			sbiz.ServiceEvalInfo{Name: "censor-evalAdsRecognition", Version: "1.0.0"},
			&middleware.ServiceFactory{
				New:      func() middleware.Service { return adses.ESet3.Kernel() },
				NewShell: func() middleware.ServiceEndpoints { return ads.EvalAdsRecognitionEndpoints{} },
			},
		)

		eacSet = set.NewEvalDirect(
			sbiz.ServiceEvalInfo{Name: "censor-evalAdsClassifier", Version: "1.0.0"},
			&middleware.ServiceFactory{
				New:      func() middleware.Service { return adses.ESet4.Kernel() },
				NewShell: func() middleware.ServiceEndpoints { return ads.EvalAdsClassifierEndpoints{} },
			},
		)
	}

	switch {
	case censorFlag <= PulpTerrorPoliticianFlag:
		// 三鉴及其中任意组合，可部署单卡，默认不操作
	case censorFlag == AdsFlag:
		// 目前ads服务占用6G显存，可独立部署一卡，默认不操作
	case censorFlag > AdsFlag && censorFlag <= AllFlag:
		// 有两个以上服务且其中包含ads，则必须分开部署
		set.UpdateInit(func() {
			pulps.AddEvalsDeployMode()
			terrors.AddEvalsDeployMode()
			politicians.AddEvalsDeployMode()
			adses.AddEvalsDeployMode()
		})
	default:
		panic("should not reach here")
	}
}
