package censor

import (
	"context"
	"encoding/json"
	"sync"
	"time"

	httputil "github.com/qiniu/http/httputil.v1"
	authstub "qiniu.com/auth/authstub.v1"

	ahttp "qiniu.com/argus/argus/com/http"
	"qiniu.com/argus/argus/com/util"
	. "qiniu.com/argus/censor/biz"
)

type ImageRequest struct {
	Datas []struct {
		DataID string `json:"data_id,omitempty"`
		URI    string `json:"uri"` // TODO 统一URI定义
	} `json:"datas"`
	Scenes []Scene `json:"scenes,omitempty"`
	Params struct {
		Scenes map[Scene]json.RawMessage `json:"scenes"`
	} `json:"params,omitempty"`
}

func (s Service) PostImageRecognition(
	ctx context.Context, req *ImageRequest, env *authstub.Env,
) (ret struct {
	Tasks []CensorResponse `json:"tasks"`
}, err error) {

	ctx, xl := util.CtxAndLog(ctx, env.W, env.Req)
	xl.Infof("post image ... %#v", req)

	// TODO 加监控

	var uid, utype uint32 = env.Uid, env.Utype

	if len(req.Scenes) == 0 {
		req.Scenes = DefaultScenes
	}

	resps := make([]CensorResponse, len(req.Datas))
	wg := sync.WaitGroup{}
	for index, _ := range req.Datas {
		wg.Add(1)
		go func(ctx context.Context, index int) {
			defer wg.Done()
			resps[index] = ImageRecognition(ctx,
				req.Datas[index].URI, s.NewImageCensorClient(uid, utype),
				req.Scenes, req.Params.Scenes,
			)
		}(util.SpawnContext(ctx), index)
	}
	wg.Wait()

	ret.Tasks = resps
	return
}

//----------------------------------------------------------------------------//

func ImageRecognition(
	ctx context.Context, uri string, cli ImageCensorClient,
	scenes []Scene, scenem map[Scene]json.RawMessage,
) CensorResponse {
	var resp = CensorResponse{
		Code: 200, Message: "OK",
		Suggestion: PASS, Scenes: map[Scene]interface{}{},
	}
	var err error

	var wg sync.WaitGroup
	var lock sync.Mutex
	for _, scene := range scenes {
		switch scene {
		case PULP:
			var params = struct {
				BlockThreshold PulpThreshold `json:"block_threshold"`
			}{}
			if scenem != nil {
				if raw, ok := scenem[PULP]; ok && raw != nil {
					_ = json.Unmarshal(raw, &params)
				}
			}
			wg.Add(1)
			go func(ctx context.Context) {
				defer wg.Done()

				req1 := ImageCensorReq{}
				req1.Data.URI = uri
				resp1, err1 := cli.PostPulp(ctx, req1)
				if err1 != nil {
					lock.Lock()
					defer lock.Unlock()
					if err == nil {
						err = err1
					}
					return
				}
				resp2 := ParseImagePulpResp(resp1.Result, params.BlockThreshold)
				lock.Lock()
				defer lock.Unlock()

				resp.Scenes[PULP] = resp2
				resp.Suggestion = resp.Suggestion.Update(resp2.Suggestion)

			}(util.SpawnContext(ctx))
		case TERROR:
			var params = struct {
				BlockThreshold TerrorThreshold `json:"block_threshold"`
			}{}
			if scenem != nil {
				if raw, ok := scenem[TERROR]; ok && raw != nil {
					_ = json.Unmarshal(raw, &params)
				}
			}
			wg.Add(1)
			go func(ctx context.Context) {
				defer wg.Done()

				req1 := ImageCensorReq{}
				req1.Data.URI = uri
				req1.Params, _ = json.Marshal(struct {
					Detail bool `json:"detail"`
				}{Detail: true})
				resp1, err1 := cli.PostTerror(ctx, req1)
				if err1 != nil {
					lock.Lock()
					defer lock.Unlock()
					if err == nil {
						err = err1
					}
					return
				}
				resp2 := ParseImageTerrorResp(resp1.Result, params.BlockThreshold)
				lock.Lock()
				defer lock.Unlock()

				resp.Scenes[TERROR] = resp2
				resp.Suggestion = resp.Suggestion.Update(resp2.Suggestion)

			}(util.SpawnContext(ctx))
		case POLITICIAN:
			var params = struct {
				BlockThreshold PoliticianThreshold `json:"block_threshold"`
			}{}
			wg.Add(1)
			go func(ctx context.Context) {
				defer wg.Done()

				req1 := ImageCensorReq{}
				req1.Data.URI = uri
				resp1, err1 := cli.PostPolitician(ctx, req1)
				if err1 != nil {
					lock.Lock()
					defer lock.Unlock()
					if err == nil {
						err = err1
					}
					return
				}
				resp2 := ParseImagePoliticianResp(resp1.Result, params.BlockThreshold)
				lock.Lock()
				defer lock.Unlock()

				resp.Scenes[POLITICIAN] = resp2
				resp.Suggestion = resp.Suggestion.Update(resp2.Suggestion)

			}(util.SpawnContext(ctx))
		}
	}

	wg.Wait()
	if err != nil {
		resp.Code, resp.Message = httputil.DetectError(err)
	}
	return resp
}

////////////////////////////////////////////////////////////////////////////////

type NewImageCensorClient func(uint32, uint32) ImageCensorClient

type ImageCensorClient interface {
	PostPulp(context.Context, ImageCensorReq) (ImageCensorPulpResp, error)
	PostTerror(context.Context, ImageCensorReq) (ImageCensorTerrorResp, error)
	PostPolitician(context.Context, ImageCensorReq) (ImageCensorPoliticianResp, error)
}

type ImageCensorReq struct {
	Data struct {
		URI string `json:"uri"`
	} `json:"data"`
	Params json.RawMessage `json:"params,omitempty"`
}

type ImageCensorPulpResp struct {
	Code    int           `json:"code"`
	Message string        `json:"message"`
	Result  ImagePulpResp `json:"result"`
}
type ImageCensorTerrorResp struct {
	Code    int             `json:"code"`
	Message string          `json:"message"`
	Result  ImageTerrorResp `json:"result"`
}
type ImageCensorPoliticianResp struct {
	Code    int                 `json:"code"`
	Message string              `json:"message"`
	Result  ImagePoliticianResp `json:"result"`
}

type imageCensorClient struct {
	Host       string
	Timeout    time.Duration
	UID, Utype uint32
}

func NewImageCensorHTTPClient(host string, timeout time.Duration) NewImageCensorClient {
	return func(uid, utype uint32) ImageCensorClient {
		return imageCensorClient{
			Host: host, Timeout: timeout,
			UID: uid, Utype: utype,
		}
	}
}

func (cli imageCensorClient) post(ctx context.Context, pth string,
	req ImageCensorReq, resp interface{},
) error {
	var (
		client = ahttp.NewQiniuStubRPCClient(cli.UID, cli.Utype, cli.Timeout)
		f      = func(ctx context.Context) error {
			return client.CallWithJson(ctx, resp, "POST", cli.Host+pth, req)
		}
	)
	return ahttp.CallWithRetry(ctx, []int{530}, []func(context.Context) error{f, f})
}

func (cli imageCensorClient) PostPulp(ctx context.Context, req ImageCensorReq) (ImageCensorPulpResp, error) {
	var resp ImageCensorPulpResp
	err := cli.post(ctx, "/v1/pulp", req, &resp)
	return resp, err
}
func (cli imageCensorClient) PostTerror(ctx context.Context, req ImageCensorReq) (ImageCensorTerrorResp, error) {
	var resp ImageCensorTerrorResp
	err := cli.post(ctx, "/v1/terror", req, &resp)
	return resp, err
}
func (cli imageCensorClient) PostPolitician(ctx context.Context, req ImageCensorReq) (ImageCensorPoliticianResp, error) {
	var resp ImageCensorPoliticianResp
	err := cli.post(ctx, "/v1/face/search/politician", req, &resp)
	return resp, err
}
