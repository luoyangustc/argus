package video

import (
	"context"
	"encoding/json"
	"sync"
	"time"

	"github.com/go-kit/kit/endpoint"
	"github.com/qiniu/rpc.v3"
	xlog "github.com/qiniu/xlog.v1"
	simage "qiniu.com/argus/AIProjects/wangan/image/wangan_mix"
	"qiniu.com/argus/AIProjects/wangan/video/wangan_mix"
	ahttp "qiniu.com/argus/argus/com/http"
	scenario "qiniu.com/argus/service/scenario/video"
	"qiniu.com/argus/service/service/image"
	vod "qiniu.com/argus/service/service/video/vod/video"
)

const (
	VERSION = "1.0.0"
)

var (
	_DOC_WANGAN_MIX = scenario.OPDoc{
		Name:    "wangan-mix",
		Version: VERSION,
		Desc:    []string{`网安场景混合图像识别`},
		Request: `
{
	...
	"ops": [
		{
			"op": "wangan_mix",
			"params": {
				"other": {
					"type": "terror"
				}
			}
		},
		...
	]
}`,
		Response: `
...
"result": {
	"label": 1,
	"score": 0.9993456,
	"classes": [
		"beheaded_isis",
		"isis_flag"
	]
}
...`,
		RequestParam: []scenario.OpDocParam{
			{Name: "ops.[].op", Type: "string", Must: "Yes", Desc: `执行的推理cmd`},
			{Name: "params.othter.type", Type: "string", Must: "No", Desc: `选择识别的类别，不同的type有不同的class，当前可选type有{"terror","internet_terror","certificate"}，默认为全部class都识别`},
		},
		ResponseParam: []scenario.OpDocParam{
			{Name: "result.label", Type: "int", Desc: "标签{0:正常，未识别出所选类别；1:识别出所选类别}"},
			{Name: "result.score", Type: "float", Desc: "识别准确度，取所有识别出的子类别中最高准确度"},
			{Name: "result.classes.[]", Type: "string", Desc: "识别出的所选子类别"},
		},
	}
)

func Import(serviceID string) func(interface{}) {
	return func(s0 interface{}) {
		s := s0.(scenario.VideoService)
		Init(s, serviceID)
	}
}

type Config struct {
	ImageHost string `json:"image_host"`
}

func (c *Config) UnmarshalJSON(b []byte) error {
	type C Config
	c2 := C{}
	if err := json.Unmarshal(b, &c2); err != nil {
		return err
	}
	*c = Config(c2)
	return nil
}

func Init(s scenario.VideoService, serviceID string) {
	var (
		config  Config
		op      wangan_mix.WanganMixOP
		once    sync.Once
		evalSet scenario.OPEvalSetter
	)

	set := vod.GetSet(s, "qiniu.com/argus/service/service/video/vod/video")
	newOP := func() {
		op = wangan_mix.NewWanganMixOP(WanganMixClient{
			client: ahttp.NewQiniuStubRPCClient(1, 0, time.Second*60),
			host:   config.ImageHost,
		})
	}
	opSet := set.RegisterOP(scenario.ServiceInfo{ID: serviceID, Version: VERSION},
		"wangan_mix", &_DOC_WANGAN_MIX, func() interface{} {
			return wangan_mix.NewOP(evalSet.Gen)
		})
	evalSet = opSet.RegisterEval(
		func() endpoint.Endpoint {
			once.Do(newOP)
			return op.NewEval()
		})
	_ = opSet.GetConfig(context.Background(), &config)
}

var _ wangan_mix.WanganMixService = WanganMixClient{}

type WanganMixClient struct {
	client *rpc.Client
	host   string
}

func (c WanganMixClient) WanganMix(ctx context.Context, req simage.WanganMixReq) (wangan_mix.WanganMixResult, error) {
	var (
		xl   = xlog.FromContextSafe(ctx)
		resp simage.WanganMixResp
	)

	var req1 struct {
		Data struct {
			URI image.STRING `json:"uri"`
		} `json:"data"`
		Params struct {
			Detail bool   `json:"detail"`
			Type   string `json:"type"`
		} `json:"params"`
	}
	req1.Data.URI = req.Data.IMG.URI
	req1.Params = req.Params
	// detail始终为true，以获取不同识别列表的score
	req1.Params.Detail = true
	err := c.client.CallWithJson(ctx, &resp, "POST", c.host+"/v1/wangan-mix", req1)
	if err != nil {
		return wangan_mix.WanganMixResult{}, err
	}
	if resp.Code != 0 && resp.Code/100 != 2 {
		xl.Warnf("wangan-mix cut failed. %#v", resp)
		return wangan_mix.WanganMixResult{}, err
	}
	ret := wangan_mix.WanganMixResult{
		Label:       resp.Result.Label,
		Score:       resp.Result.Score,
		Classes:     resp.Result.Classes,
		ClassScores: make(map[string]float32, 0),
	}
	if resp.Result.Label == 1 {
		for _, cl := range resp.Result.Classify {
			if cl.Class != "normal" && cl.Score > ret.ClassScores[cl.Class] {
				ret.ClassScores[cl.Class] = cl.Score
			}
		}
		for _, dt := range resp.Result.Detection {
			if dt.Score > ret.ClassScores[dt.Class] {
				ret.ClassScores[dt.Class] = dt.Score
			}
		}
	} else {
		ret.ClassScores["normal"] = resp.Result.Score
	}
	return ret, nil
}
