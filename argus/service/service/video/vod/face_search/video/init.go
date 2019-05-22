package video

import (
	"context"
	"encoding/json"
	"sync"
	"time"

	"github.com/go-kit/kit/endpoint"

	rpc "github.com/qiniu/rpc.v3"
	xlog "github.com/qiniu/xlog.v1"
	ahttp "qiniu.com/argus/argus/com/http"
	scenario "qiniu.com/argus/service/scenario/video"
	"qiniu.com/argus/service/service/image"
	"qiniu.com/argus/service/service/video/vod/face_search"
	vod "qiniu.com/argus/service/service/video/vod/video"
)

const (
	VERSION = "1.0.0"
)

var (
	_DOC_FACE_SEARCH = scenario.OPDoc{
		Name:    "face_group_search_private",
		Version: VERSION,
		Desc:    []string{`人脸1：N搜索`},
		Request: `
{
	...
	"ops": [
		{
			"op": "face_group_search_private",
			"cut_hook_url": "http://yy.com/yyy",
			"params": {
				"other": {
					"groups": ["test_group_001","test_group_002"],
					"cluster": "test_cluster_001" 
					"threshold": 0.35,
					"limit": 1
				} 
			}
		},
		...
	]
}`,
		Response: `
{
	...
	"result": {
		"faces": [
			{
				"bounding_box": {
					"pts": [[606,515],[649,515],[649,576],[606,576]],
					"score":0.9916495					
				},
				"faces":[
					{
						"id":"tkLwryRJ0vUT2-ZYZO-dCQ==",
						"score":0.6523004,
						"tag":"张三",
						"desc":""
					}
					...
				]
			}
			...
		]
	}
	...
}`,
		RequestParam: []scenario.OpDocParam{
			{Name: "ops.[].op", Type: "string", Must: "Yes", Desc: `执行的推理cmd`},
			{Name: "ops.[].cut_hook_url", Type: "string", Must: "No", Desc: `截帧回调地址`},
			{Name: "ops.[].params.other.groups", Type: "string array", Must: "No", Desc: `人脸搜索库列表，独立标识人脸库，不能与排重库同时为空`},
			{Name: "ops.[].params.other.cluster", Type: "string", Must: "No", Desc: `人脸排重库，不能与搜索库列表同时为空`},
			{Name: "ops.[].params.other.threshold", Type: "float", Must: "No", Desc: `人脸搜索相似度阈值，即相似度高于该值才返回结果，默认为0,推荐0.35`},
			{Name: "ops.[].params.other.limit", Type: "int", Must: "No", Desc: `相似人脸项数目限制，范围[1,20],按相似度降序排列，默认limit=1`},
		},
		ResponseParam: []scenario.OpDocParam{
			{Name: "faces", Type: "array", Desc: `该帧中检测出的人脸列表`},
			{Name: "faces.[].bounding_box.pts", Type: "array", Desc: `人脸所在图片中的位置，四点坐标值[左上，右上，右下，左下] 四点坐标框定的脸部`},
			{Name: "faces.[].bounding_box.score", Type: "float", Desc: `人脸的检测置信度人脸的检测置信度`},
			{Name: "faces.[].faces", Type: "array", Desc: `和此人脸相似的人脸的列表`},
			{Name: "faces.[].faces.[].id", Type: "string", Desc: `相似人脸ID`},
			{Name: "faces.[].faces.[].score", Type: "float", Desc: `两个人脸的相似度`},
			{Name: "faces.[].faces.[].tag", Type: "string", Desc: `相似人脸Tag`},
			{Name: "faces.[].faces.[].desc", Type: "string", Desc: `相似人脸描述`},
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
	var set = vod.GetSet(s, "qiniu.com/argus/service/service/video/vod/video")
	var config = Config{}

	var ss face_search.FaceSearchOP
	var once sync.Once
	newSS := func() {
		ss = face_search.NewFaceSearchOP(Face_SearchClient{
			client: ahttp.NewQiniuStubRPCClient(1, 0, time.Second*60),
			host:   config.ImageHost,
		})
	}

	var evalSet scenario.OPEvalSetter

	opSet := set.RegisterOP(scenario.ServiceInfo{ID: serviceID, Version: VERSION},
		"face_group_search_private", &_DOC_FACE_SEARCH, func() interface{} {
			return face_search.NewOP(evalSet.Gen)
		})
	evalSet = opSet.RegisterEval(func() endpoint.Endpoint {
		once.Do(newSS)
		return ss.NewEval()
	})

	_ = opSet.GetConfig(context.Background(), &config)

}

var _ face_search.FaceSearchService = Face_SearchClient{}

type Face_SearchClient struct {
	client *rpc.Client
	host   string
}

func (c Face_SearchClient) FaceSearch(ctx context.Context, args face_search.FaceSearchReq) (face_search.FaceSearchResp, error) {
	var (
		xl   = xlog.FromContextSafe(ctx)
		resp face_search.FaceSearchResp
	)

	err := c.client.CallWithJson(ctx, &resp, "POST", c.host+"/v1/face/groups/multi/search",
		struct {
			Images       []image.STRING `json:"images"`
			Groups       []string       `json:"groups"`
			ClusterGroup string         `json:"cluster_group"`
			Threshold    float32        `json:"threshold"`
			Limit        int            `json:"limit"`
		}{
			Images:       []image.STRING{args.Data.IMG.URI},
			Groups:       args.Params.Groups,
			ClusterGroup: args.Params.Cluster,
			Threshold:    args.Params.Threshold,
			Limit:        args.Params.Limit,
		},
	)
	if err != nil {
		return face_search.FaceSearchResp{}, err
	}
	if resp.Code != 0 && resp.Code/100 != 2 {
		xl.Warnf("face_search cut failed. %#v", resp)
		return face_search.FaceSearchResp{}, err
	}
	return resp, nil

}
