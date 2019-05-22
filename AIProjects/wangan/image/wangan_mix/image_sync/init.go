package image_sync

import (
	"context"
	"encoding/json"

	"github.com/go-kit/kit/endpoint"
	"github.com/imdario/mergo"
	"qiniu.com/argus/AIProjects/wangan/image/wangan_mix"
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
	_DOC_WANGAN_MIX = scenario.APIDoc{
		Version: VERSION,
		Desc:    []string{`网安场景混合图像识别`},
		Request: `POST /v1/wangan-mix Http/1.1
Content-Type:application/json

{
	"data": {
		"uri": "http://image2.jpeg"
	},
	"params": {
		"detail": true,
		"type": "terror"
	}
}`,
		Response: `200 ok
Content-Type:application/json

{
	"code": ,
	"message": "",
	"result": {
		"label": 1,
		"score": 0.99917644,
		"classes": ["march_banner","tibetan_flag"],
		"classify": [
            {
                "class": "march_banner",
                "score": 0.99917644
            }
        ],
		"detection": [
			{
                "class": "tibetan_flag",
                "score": 0.9248621,
                "pts": [[175,114], [222,114], [222,191], [175,191]]
            }
		]
	}
}`,
		RequestParam: []scenario.APIDocParam{
			{Name: "data.uri", Type: "string", Desc: "图片资源地址"},
			{Name: "params.detail", Type: "bool", Desc: "是否显示混合识别详细结果，默认为false"},
			{Name: "params.type", Type: "string", Desc: `选择识别的类别，不同的type有不同的class，当前可选type有{"terror","internet_terror","certificate"}，默认为全部class都识别`},
		},
		ResponseParam: []scenario.APIDocParam{
			{Name: "code", Type: "int", Desc: "0:表示处理成功；不为0:表示出错"},
			{Name: "message", Type: "string", Desc: "描述结果或出错信息"},
			{Name: "result.label", Type: "int", Desc: "标签{0:正常，未识别出所选类别；1:识别出所选类别}"},
			{Name: "result.score", Type: "float", Desc: "识别准确度，取所有识别出的子类别中最高准确度"},
			{Name: "result.classes.[]", Type: "string", Desc: "图片中识别出的所选子类别"},
			{Name: "result.classify.[].class", Type: "string", Desc: "分类识别的类别名称(当detail=true时返回)"},
			{Name: "result.classify.[].score", Type: "float", Desc: "分类识别的准确度(当detail=true时返回)"},
			{Name: "result.detection.[].class", Type: "string", Desc: "检测识别的类别名称(当detail=true时返回)"},
			{Name: "result.detection.[].score", Type: "float", Desc: "检测识别的准确度(当detail=true时返回)"},
			{Name: "result.detection.[].pts", Type: "list", Desc: "检测识别的物体在图片中的位置，四点坐标值 [左上，右上，右下，左下] 四点坐标框定的物体区域(当detail=true时返回)"},
		},
		ErrorMessage: []scenario.APIDocError{},
		Appendix: []string{`**混合图像识别标签取值即说明：**

| 标签 | 说明 |
| :--- | :--- |
| knives_true | 真刀 |
| knives_false | 动漫刀 |
| knives_kitchen | 厨房刀 |
| guns_true | 真枪 |
| guns_anime | 动漫枪 |
| guns_tools | 玩具枪 |
| BK_LOGO_1 | 暴恐_ISIS_1号_台标 |
| BK_LOGO_2 | 暴恐_ISIS_2号_台标 |
| BK_LOGO_3 | 暴恐_ISIS_3号_台标 |
| BK_LOGO_4 | 暴恐_ISIS_4号_台标 |
| BK_LOGO_5 | 暴恐_ISIS_5号_台标 |
| BK_LOGO_6 | 暴恐_ISIS_6号_台标 |
| isis_flag | ISIS旗帜 |
| islamic_flag | 伊斯兰教旗帜 |
| tibetan_flag | 藏独旗帜 |
| falungong_logo | 法论功标志 |
| idcard_positive | 身份证正面 |
| idcard_negative | 身份证背面 |
| bankcard_positive | 银行卡正面 |
| bankcard_negative | 银行卡背面 |
| gongzhang | 公章 |
| bloodiness_human | 血腥_人物血腥 |
| bomb_fire | 爆炸_明火爆炸 |
| bomb_smoke | 爆炸_烟雾爆炸 |
| bomb_vehicle | 爆炸_汽车爆炸 |
| bomb_self-burning	| 爆炸_人体自焚 |
| beheaded_isis | 斩首_恐怖分子 |
| beheaded_decollation | 斩首_残酷行刑 |
| march_banner | 游行_抗议横幅 |
| march_crowed | 游行_非法集会 |
| fight_police | 斗殴_警民冲突 |
| fight_person | 斗殴_打架斗殴 |
| character | 敏感_字符文字 |
| masked | 敏感_蒙面 |
| army | 敏感_战争军队 |
| scene_person | 敏感_人物 |
| anime_likely_bloodiness | 敏感_血腥类_动漫 |
| anime_likely_bomb | 敏感_爆炸类_动漫 |
| islamic_dress | 敏感_着装 |

`,
			wangan_mix.MIX_CLASSES.Doc()},
	}
)

type Config wangan_mix.Config

func (c *Config) UnmarshalJSON(b []byte) error {
	type C Config
	c2 := C{}
	if err := json.Unmarshal(b, &c2); err != nil {
		return err
	}
	_ = mergo.Merge(&c2, C(wangan_mix.DEFAULT))
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
		config = Config(wangan_mix.DEFAULT)
		eSet   biz.ServiceEvalSetter
	)

	set := is.Register(
		scenario.ServiceInfo{ID: serviceID, Name: "wangan-mix", Version: VERSION},
		&middleware.ServiceFactory{
			New: func() middleware.Service {
				es, _ := eSet.Gen()
				eval := es.(wangan_mix.EvalWanganMixService)
				s, _ := wangan_mix.NewWanganMixService(wangan_mix.Config(config), eval)
				return s
			},
			NewShell: func() middleware.ServiceEndpoints { return wangan_mix.WanganMixEndPoints{} },
		},
	).UpdateRouter(func(
		sf func() middleware.Service,
		imageParser func() pimage.IImageParse,
		path func(string) *scenario.ServiceRoute,
	) error {
		endp := func() wangan_mix.WanganMixEndPoints {
			svc := sf()
			endp, ok := svc.(wangan_mix.WanganMixEndPoints)
			if !ok {
				svc, _ := middleware.MakeMiddleware(svc, wangan_mix.WanganMixEndPoints{}, nil, nil)
				endp = svc.(wangan_mix.WanganMixEndPoints)
			}
			return endp
		}

		type WanganMixReq struct {
			Data struct {
				URI string `json:"uri"`
			} `json:"data"`
			Params struct {
				Detail bool   `json:"detail"`
				Type   string `json:"type"`
			} `json:"params"`
		}
		path("/v1/wangan-mix").Doc(_DOC_WANGAN_MIX).Route().Methods("POST").Handler((transport.MakeHttpServer(
			func(ctx context.Context, req0 interface{}) (interface{}, error) {
				var (
					req1 wangan_mix.WanganMixReq
					err  error
				)
				req2, _ := req0.(WanganMixReq)
				req1.Data.IMG, err = imageParser().ParseImage(ctx, req2.Data.URI)
				req1.Params = req2.Params
				if err != nil {
					return nil, err
				}
				return endp().WanganMixEP(ctx, req1)
			},
			WanganMixReq{},
		)))

		return nil
	})

	_ = set.GetConfig(context.Background(), &config)

	eSet = set.NewEval(
		biz.ServiceEvalInfo{Name: "evalWanganMix", Version: "1.0.0"},
		WanganMixEvalClient,
		func() middleware.ServiceEndpoints { return wangan_mix.EvalWanganMixEndPoints{} },
	).SetModel(sbiz.EvalModelConfig{
		Image: "reg.qiniu.com/avaprd/aisdk-wa_20181207:20190128-v198-ATLAB-10169",
		CustomFiles: map[string]string{
			"fine_weight.bin": "wa-20181207/tensorrt/wangan_v1.2_20190123/fine_weight.bin",
			"fine_labels.csv": "wa-20181207/tensorrt/wangan_v1.2_20190123/classify_labels.csv",
			"det_weight.bin":  "wa-20181207/tensorrt/wangan_v1.2_20190123/det_weight.bin",
			"det_labels.csv":  "wa-20181207/tensorrt/wangan_v1.2_20190123/det_labels.csv",
		},
		Args: &sbiz.ModelConfigArgs{
			BatchSize: 4,
			CustomValues: map[string]interface{}{
				"gpu_id":   0,
				"frontend": "ipc://frontend.ipc",
				"backend":  "ipc://backend.ipc",
			},
		},
		Type: sbiz.EvalRunTypeServing,
	})
}

func WanganMixEvalClient(
	client func(method string, respSample interface{}) (endpoint.Endpoint, error),
) middleware.Service {
	end, _ := client("POST", wangan_mix.EvalWanganMixResp{})
	return wangan_mix.EvalWanganMixEndPoints{
		EvalWanganMixEP: func(ctx context.Context, req0 interface{}) (interface{}, error) {
			type WanganMixReq struct {
				Data struct {
					URI string `json:"uri"`
				} `json:"data"`
				Params struct {
					Type   string `json:"type"`
					Detail bool   `json:"detail"`
				} `json:"params"`
			}
			req1, _ := req0.(wangan_mix.EvalWanganMixReq)
			var req2 WanganMixReq
			req2.Data.URI = string(req1.Data.IMG.URI)
			return end(ctx, req2)
		}}
}
