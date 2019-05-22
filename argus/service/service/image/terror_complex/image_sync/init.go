package image_sync

import (
	"context"
	"encoding/json"

	"github.com/imdario/mergo"

	"qiniu.com/argus/service/middleware"
	sbiz "qiniu.com/argus/service/scenario/biz"
	scenario "qiniu.com/argus/service/scenario/image_sync"
	pimage "qiniu.com/argus/service/service/image"
	"qiniu.com/argus/service/service/image/terror"
	timagesync "qiniu.com/argus/service/service/image/terror/image_sync"
	"qiniu.com/argus/service/service/image/terror_complex"
	"qiniu.com/argus/service/transport"
)

var ON bool = false

const (
	VERSION = "1.1.0"
)

var (
	_DOC_TERROR_COMPLEX = scenario.APIDoc{
		Version: VERSION,
		Desc:    []string{`用检测暴恐识别和分类暴恐识别方法做融合暴恐识别`, `每次输入一张图片，返回其内容是否含暴恐信息`},
		Request: `POST /v1/terror/complex  Http/1.1
Content-Type: application/json

{
	"data": {
		"uri": "http://xxx/xxx.jpg"
	},
	"params": {
		"detail": true
	}
}`,
		Response: ` 200 ok
Content-Type:application/json

{
	"code": 0,
	"message": "",
	"result": {
		"label":1,
		"score":0.987,
		"review":false
		"classes": [
			{
				"class":"bomb_fire",
				"score": 0.97
			},
			{
				"class": "guns",
				"score": 0.95
			}]
	}
}`,
		RequestParam: []scenario.APIDocParam{
			{Name: "uri", Type: "string", Desc: "图片资源地址"},
			{Name: "params.detail", Type: "bool", Desc: "是否显示详细信息；可选参数"},
		},
		ResponseParam: []scenario.APIDocParam{
			{Name: "code", Type: "int", Desc: "0:表示正确"},
			{Name: "message", Type: "string", Desc: "结果描述信息"},
			{Name: "result.label", Type: "int", Desc: "标签{0:正常，1:暴恐}"},
			{Name: "result.classes.class", Type: "string", Desc: "标签类别（指定detail=true的情况下返回）"},
			{Name: "result.classes.score", Type: "float32", Desc: "标签类别准确度（指定detail=true的情况下返回）"},
			{Name: "result.score", Type: "float", Desc: "暴恐识别准确度，取所有标签中最高准确度"},
			{Name: "result.review", Type: "bool", Desc: "是否需要人工review"}},
		ErrorMessage: []scenario.APIDocError{},
		Appendix: []string{`暴恐标签取值及说明：

| 标签 | 说明 |
| :--- | :--- |
| islamic flag | 伊斯兰教旗帜 |
| isis flag | ISIS旗帜 |
| tibetan flag | 藏独旗帜 |
| knives | 刀 |
| guns | 枪 |
| bloodiness_human | 人物血腥 |
| bomb_fire | 明火爆炸 |
| bomb_smoke | 烟雾爆炸 |
| bomb_vehicle | 汽车爆炸 |
| bomb_self-burning | 人体自焚 |
| beheaded_isis | 斩首：恐怖分子 |
| beheaded_decollation | 斩首：残忍行刑 |
| march_banner | 抗议横幅 |
| march_crowed | 非法集会 |
| fight_police | 警民冲突 |
| fight_person | 打架斗殴 |
| character | 敏感字符文字 |
| masked | 蒙面 |
| army | 战争军队 |
| scene_person | 敏感人物 |
| anime_likely_bloodiness | 二次元血腥 |
| anime_likely_bomb | 二次元爆炸 |
| islamic_dress | 敏感着装 |
| normal | 正常 |
`},
	}
)

type Config terror_complex.Config

func (c *Config) UnmarshalJSON(b []byte) error {
	type C Config
	c2 := C{}
	if err := json.Unmarshal(b, &c2); err != nil {
		return err
	}
	c1 := C(terror_complex.DEFAULT)
	_ = mergo.Merge(&c2, &c1)
	*c = Config(c2)
	return nil
}

func Import(serviceID string) func(interface{}) {
	return func(is interface{}) {
		Init(is.(scenario.ImageServer), serviceID)
	}
}

func Init(is scenario.ImageServer, serviceID string) {
	var config = Config(terror_complex.DEFAULT)

	var ts1 terror.EvalTerrorMixupService
	var ts2 terror.EvalTerrorDetectService

	var eSet1 sbiz.ServiceEvalSetter
	var eSet2 sbiz.ServiceEvalSetter

	ON = true
	set := is.Register(
		scenario.ServiceInfo{ID: serviceID, Name: "terror_complex", Version: VERSION},
		&middleware.ServiceFactory{
			New: func() middleware.Service {
				{
					ss1, _ := eSet1.Gen()
					ts1 = ss1.(terror.EvalTerrorMixupService)

					ss2, _ := eSet2.Gen()
					ts2 = ss2.(terror.EvalTerrorDetectService)

				}
				s, _ := terror_complex.NewTerrorComplexService(terror_complex.Config(config), ts2, ts1)
				return s
			},
			NewShell: func() middleware.ServiceEndpoints { return terror_complex.TerrorComplexEndpoints{} },
		}).
		UpdateRouter(func(
			sf func() middleware.Service,
			imageParser func() pimage.IImageParse,
			path func(string) *scenario.ServiceRoute,
		) error {
			endp := func() terror_complex.TerrorComplexEndpoints {
				svc := sf()
				endp, ok := svc.(terror_complex.TerrorComplexEndpoints)
				if !ok {
					svc, _ = middleware.MakeMiddleware(svc, terror_complex.TerrorComplexEndpoints{}, nil, nil)
					endp = svc.(terror_complex.TerrorComplexEndpoints)
				}
				return endp
			}
			type Req struct {
				Data struct {
					URI string `json:"uri"`
				} `json:"data"`
				Params struct {
					Detail bool `json:"detail"`
				} `json:"params"`
			}
			path("/v1/terror/complex").Doc(_DOC_TERROR_COMPLEX).Route().
				Methods("POST").Handler(transport.MakeHttpServer(
				func(ctx context.Context, req0 interface{}) (interface{}, error) {
					req1, _ := req0.(Req)
					var req2 terror.TerrorReq
					var err error
					req2.Data.IMG, err = imageParser().ParseImage(ctx, req1.Data.URI)
					if err != nil {
						return nil, err
					}
					req2.Params.Detail = req1.Params.Detail
					return endp().TerrorComplexEP(ctx, req2)
				},
				Req{}))
			return nil
		})

	_ = set.GetConfig(context.Background(), &config)

	eSet1 = set.NewEval(
		sbiz.ServiceEvalInfo{Name: "evalTerrorMixup", Version: "1.1.0"},
		timagesync.TerrorMixupEvalClient,
		func() middleware.ServiceEndpoints { return terror.EvalTerrorMixupEndpoints{} },
	).SetModel(terror_complex.EVAL_TERROR_MIXUP_CONFIG).GenId()

	eSet2 = set.NewEval(
		sbiz.ServiceEvalInfo{Name: "evalTerrorDetect", Version: "1.1.0"},
		timagesync.TerrorDetEvalClient,
		func() middleware.ServiceEndpoints { return terror.EvalTerrorDetectEndpoints{} },
	).SetModel(terror_complex.EVAL_TERROR_DET_CONFIG).GenId()

}
