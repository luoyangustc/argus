package video

import (
	"context"
	"encoding/json"
	"net/http"
	"path"
	"sync"

	"github.com/gorilla/mux"
	"github.com/imdario/mergo"

	restrpc "github.com/qiniu/http/restrpc.v1"
	xlog "github.com/qiniu/xlog.v1"

	"qiniu.com/argus/service/middleware"
	scenario "qiniu.com/argus/service/scenario/video"
	ads "qiniu.com/argus/service/service/image/ads/image_sync"
	icensor "qiniu.com/argus/service/service/image/censor/image_sync"
	politician "qiniu.com/argus/service/service/image/politician/image_sync"
	pulp "qiniu.com/argus/service/service/image/pulp/image_sync"
	terror "qiniu.com/argus/service/service/image/terror/image_sync"
	svideo "qiniu.com/argus/service/service/video"
	"qiniu.com/argus/service/service/video/censor"
	"qiniu.com/argus/service/transport"
	"qiniu.com/argus/video/vframe"
)

const (
	PULP       string = "pulp"
	TERROR     string = "terror"
	POLITICIAN string = "politician"
	ADS        string = "ads"
)

var (
	OP_FLAG = map[string]int{
		PULP:       icensor.PulpFlag,
		TERROR:     icensor.TerrorFlag,
		POLITICIAN: icensor.PoliticianFlag,
		ADS:        icensor.AdsFlag,
	}

	_DEFAULT_CENSOR_CONFIG = Config{
		CutParam: censor.CutParam{
			Mode:          0,
			IntervalMsecs: 5000,
		},
		SaveParam: &svideo.MinioConfig{
			Host:   "127.0.0.1:9888",
			AK:     "admin",
			SK:     "minio-12345",
			Bucket: "public",
			Prefix: "censor",
		},
	}

	_DOC_SYNC_CENSOR_VIDEO = scenario.APIDoc{
		Name:    `同步视频审核`,
		Version: "1.1.0",
		Desc:    []string{`同步视频审核分析`},
		Request: `POST /v3/censor/video
Content-Type: application/json

{
	"data": {
		"uri": "http://xx"
	},
	"params": {
		"scenes": [
			<scene:string>,
			...
		]
		"cut_param": {
			"interval_msecs": <interval_msecs:int>
		}
	}
}`,
		Response: `HTTP/1.1 200 OK
Content-Type: application/json

{
	"code": 200,
	"message": "OK",
	"result": {
		"suggestion": <suggestion:string>,
		"scenes": {
			<scene:string>: {
				"suggestion": <suggestion:string>,
				"cuts": [
					{
						"suggestion": <suggestion:string>,
						"offset": <offset:int>,
						"details": [
							{
								"suggestion": <suggestion:string>,
								"label": <label:string>,
								"group": <group:string>,
								"score": <score:float>,
								"detections": [
									{
										"pts": <pts:[4][2]int>
										"score": <score:float>,
									},
									...
								]
							},
							...
						]
					},
					...
				]
			},
			...
		}
	}
}`,
		RequestParam: []scenario.APIDocParam{
			{Name: "data.uri", Type: "string", Must: "Yes", Desc: `视频地址，支持http和https`},
			{Name: "params.scenes", Type: "list", Must: "Yes", Desc: `审核类型，目前支持"pulp", "terror", "politician", "ads"`},
			{Name: "params.cut_param.interval_msecs", Type: "int", Must: "No", Desc: `截帧频率，单位：毫秒，默认为5000（5s)，取值范围为1000～60000（即1s~60s）。`},
		},
		ResponseParam: []scenario.APIDocParam{
			{Name: "code", Type: "int", Desc: `处理状态：200 调用成功`},
			{Name: "message", Type: "string", Desc: `与code对应的状态描述信息`},
			{Name: "result.suggestion", Type: "string", Desc: `视频的总审核结果`},
			{Name: "result.scenes", Type: "map", Desc: `各个审核类型结果，与请求参数scenes对应`},
			{Name: "result.scenes.<scene>.suggestion", Type: "string", Desc: `该审核类型的总审核结果`},
			{Name: "result.scenes.<scene>.cuts", Type: "list", Desc: `截帧审核结果数组`},
			{Name: "result.scenes.<scene>.cuts.[].suggestion", Type: "string", Desc: `该审核类型下的该截帧的审核结果`},
			{Name: "result.scenes.<scene>.cuts.[].offset", Type: "int", Desc: `截帧的时间位置，单位毫秒`},
			{Name: "result.scenes.<scene>.cuts.[].details", Type: "list", Desc: `截帧的详细信息数组`},
			{Name: "result.scenes.<scene>.cuts.[].details.[].suggestion", Type: "string", Desc: `该详细信息的审核结果`},
			{Name: "label", Type: "string", Desc: `该详细信息的标签，具体含义取决于审核类型，见附录`},
			{Name: "group", Type: "string", Desc: `该详细信息的分组，该字段目前只在"politician"类型中返回`},
			{Name: "score", Type: "float", Desc: `该详细信息的结果置信度`},
			{Name: "detections", Type: "list", Desc: `该详细信息在图片中出现的检测框位置列表，该字段只在如下审核类型中返回：1. politician；2. terror下的部分label。见附录`},
			{Name: "detections.[].pts", Type: "[][]int", Desc: `检测到的物体坐标框，四点坐标值[左上，右上，右下，左下]`},
			{Name: "detections.[].score", Type: "float", Desc: `检测到的物体置信度`},
			{Name: "detections.[].comments", Type: "float", Desc: `广告审核，当审核结果label为"ads"时，该字段用于返回识别到的敏感词`},
		},
		Appendix: []string{`审核类型pulp：label字段取值及说明：

| 标签 | 说明 |
| :--- | :--- |
| pulp | 色情 |
| sexy | 性感 |
| normal | 正常 |
`, `审核类型terror：label字段取值及说明：

| 标签 | 说明 | 是否返回detections字段 |
| :--- | :--- | :--- |
| illegal_flag | 违规旗帜 | Yes |
| knives | 刀 | Yes |
| guns | 枪 | Yes |
| anime_knives | 二次元刀 | Yes |
| anime_guns | 二次元枪 | Yes |
| bloodiness | 血腥 | No |
| bomb | 爆炸 | No |
| self_burning | 自焚 | No |
| beheaded | 行刑斩首 | No |
| march_crowed | 非法集会 | No |
| fight_police | 警民冲突 | No |
| fight_person | 打架斗殴 | No |
| army | 军队 | No |
| special_characters | 特殊字符 | No |
| anime_bloodiness | 二次元血腥 | No |
| special_clothing | 特殊着装 | No |
| bloodiness_animal | 动物血腥 | No |
| fire_weapon |  武器发射火焰 | No |
| normal | 正常 | No |
`, `审核类型politician：label字段为政治人物姓名，group字段取值及说明如下：

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
`, `审核类型ads：label字段取值及说明：

| 标签 | 说明 |
| :--- | :--- |
| qr_code | 二维码 |
| bar_code | 条形码 |
| ads | 广告详情，该标签下会返回检测到的广告坐标框和其中的敏感词 |
| summary_ads | 整体广告总结 |
`},
	}
)

type Config struct {
	CutParam  censor.CutParam     `json:"default_cut_param"`
	SaveParam *svideo.MinioConfig `json:"default_save_param"`
}

func (c *Config) UnmarshalJSON(b []byte) error {
	type C Config
	c2 := C{}
	if err := json.Unmarshal(b, &c2); err != nil {
		return err
	}
	c1 := C(_DEFAULT_CENSOR_CONFIG)
	_ = mergo.Merge(&c2, &c1)
	*c = Config(c2)
	return nil
}

type Set struct {
	Config
	scenario.ServiceSetter
	censor.Service
	censor.OPs
	vframe.URIProxy
}

var once sync.Once
var set *Set

func Init(s scenario.VideoService, serviceID string) {
	set := GetSet(s, serviceID)
	_ = set
}

func GetSet(s scenario.VideoService, serviceID string) *Set {
	once.Do(func() {
		set = &Set{Config: _DEFAULT_CENSOR_CONFIG}
		set.ServiceSetter = s.Register(
			scenario.ServiceInfo{ID: serviceID, Version: "0.8.0"},
			func(_ops svideo.OPs) error {
				var censorFlag int
				set.OPs = make(censor.OPs)
				for op, of := range _ops {
					set.OPs[op] = of.(censor.OPFactory)
					censorFlag |= OP_FLAG[op]
				}

				switch {
				case censorFlag <= icensor.PulpTerrorPoliticianFlag:
					// 三鉴及其中任意组合，可部署单卡，默认不操作
				case censorFlag == icensor.AdsFlag:
					// 目前ads服务占用6G显存，可独立部署一卡，默认不操作
				case censorFlag > icensor.AdsFlag && censorFlag <= icensor.AllFlag:
					// 有两个以上服务且其中包含ads，则必须分开部署
					for op, _ := range _ops {
						opSetter := set.GetOP(op)
						switch op {
						case PULP:
							opSetter.AddEvalsDeployModeOnGPU("", pulp.DeployMode2GPUCard)
							opSetter.AddEvalsDeployModeOnGPU("", pulp.DeployMode4GPUCard)
						case TERROR:
							opSetter.AddEvalsDeployModeOnGPU("", terror.DeployMode2GPUCard)
							opSetter.AddEvalsDeployModeOnGPU("", terror.DeployMode4GPUCard)
						case POLITICIAN:
							opSetter.AddEvalsDeployModeOnGPU("", politician.DeployMode2GPUCard)
							opSetter.AddEvalsDeployModeOnGPU("", politician.DeployMode4GPUCard)
						case ADS:
							opSetter.AddEvalsDeployModeOnGPU("", ads.DeployMode2GPUCard)
							opSetter.AddEvalsDeployModeOnGPU("", ads.DeployMode4GPUCard)
						default:
						}
					}

				default:
					panic("should not reach here")
				}

				return nil
			},
			func() {
				var (
					ctx       = context.Background()
					xl        = xlog.FromContextSafe(ctx)
					saverHook svideo.SaverHook
					err       error
				)

				if set.SaveParam != nil {
					saverHook, err = svideo.NewMinioSaver(*set.SaveParam)
					if err != nil {
						xl.Warnf("fail to connect to file server: %#v, %v", set.SaveParam, err)
					}
				}

				set.URIProxy = vframe.NewURIProxy(
					"http://127.0.0.1:" + set.Router().Port + "/uri")

				set.Service = censor.NewService(
					ctx,
					set.CutParam,
					saverHook,
					set.URIProxy,
					path.Join(set.Workspace(), "data"),
					set.OPs,
				)
			},
		)
		_ = set.GetConfig(context.Background(), &set.Config)
		_ = sync_(set)
	})
	return set
}

func sync_(set *Set) error {

	rs := set.Router().NewRouteSetter("censor",
		func() interface{} {
			return set.Service
		},
	)
	rs.UpdateRouter(func(
		sf func() middleware.Service,
		newVS func() interface{},
		path func(string) *scenario.Route,
	) error {
		path("/v3/censor/video").Doc(_DOC_SYNC_CENSOR_VIDEO).Route().Methods("POST").Handler(transport.MakeHttpServer(
			func(ctx context.Context, req0 interface{}) (interface{}, error) {
				req := req0.(censor.VideoCensorReq)
				return newVS().(censor.Service).Video(ctx, req)
			},
			censor.VideoCensorReq{}))

		return nil
	})
	set.Router().Path("/uri/{uri}").HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		vars := mux.Vars(r)
		set.URIProxy.GetUri_(context.Background(),
			&struct {
				CmdArgs []string
			}{CmdArgs: []string{vars["uri"]}},
			&restrpc.Env{W: w, Req: r},
		)
	})

	return nil
}
