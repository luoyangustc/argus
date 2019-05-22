package censor

import (
	"context"
	"encoding/json"
	"sync"

	xlog "github.com/qiniu/xlog.v1"

	"qiniu.com/argus/com/util"
	. "qiniu.com/argus/service/service"
	pimage "qiniu.com/argus/service/service/image"
)

const (
	PULP       string = "pulp"
	TERROR     string = "terror"
	POLITICIAN string = "politician"
	ADS        string = "ads"
)

type IPremierCensorRequest struct {
	Data struct {
		DataID string       `json:"data_id,omitempty"`
		IMG    pimage.Image `json:"-"`
		URI    string       `json:"uri"`
	} `json:"data"`
	Datas []struct {
		DataID string       `json:"data_id,omitempty"`
		IMG    pimage.Image `json:"-"`
		URI    string       `json:"uri"`
	} `json:"datas"`
	Params struct {
		Scenes       []string                   `json:"scenes,omitempty"`
		ScenesParams map[string]json.RawMessage `json:"scenes_params"`
	} `json:"params,omitempty"`
}

type CensorResponse struct {
	Code    int          `json:"code"`
	Message string       `json:"message"`
	Result  CensorResult `json:"result"`
}

type CensorResult struct {
	Suggestion pimage.Suggestion             `json:"suggestion"` // 审核结论
	Scenes     map[string]pimage.SceneResult `json:"scenes,omitempty"`
}

type IPremierCensorResp struct {
	Tasks []CensorResult `json:"tasks"`
}

func (s censorService) PremierCensor(ctx context.Context, req IPremierCensorRequest) (ret CensorResponse, err error) {

	xl := xlog.FromContextSafe(ctx)
	xl.Infof("post image ... %v, %v", req.Data.URI, req.Params)

	// check scenes
	if len(req.Params.Scenes) == 0 {
		xl.Warnf("empty scenes")
		return CensorResponse{}, ErrArgs("empty scene")
	}
	for _, sc := range req.Params.Scenes {
		if _, ok := s.scenes[sc]; !ok {
			xl.Warnf("bad scene: %v", sc)
			return CensorResponse{}, ErrArgs("bad scene")
		}
	}

	ret.Result, err = s.ImageRecognition(ctx, req.Data.IMG, req.Params.Scenes, req.Params.ScenesParams)
	if err != nil {
		return CensorResponse{}, err
	}

	ret.Code = 200
	ret.Message = "OK"
	return
}

func (s censorService) ImageRecognition(
	ctx context.Context, uri pimage.Image,
	scenes []string, scenem map[string]json.RawMessage,
) (CensorResult, error) {
	var resp = CensorResult{
		Suggestion: pimage.PASS,
		Scenes:     map[string]pimage.SceneResult{},
	}
	var err error

	var wg sync.WaitGroup
	var lock sync.Mutex
	for _, scene := range scenes {
		switch scene {
		case PULP:
			wg.Add(1)
			go func(ctx context.Context) {
				defer wg.Done()

				req1 := pimage.ImageCensorReq{}
				req1.Data.IMG = uri
				req1.Params = scenem[PULP]
				resp1, err1 := s.PulpCensor(ctx, req1)
				if err1 != nil {
					lock.Lock()
					defer lock.Unlock()
					if err == nil {
						err = err1
					}
					return
				}
				lock.Lock()
				defer lock.Unlock()

				resp.Scenes[PULP] = resp1
				resp.Suggestion = resp.Suggestion.Update(resp1.Suggestion)

			}(util.SpawnContext(ctx))
		case TERROR:
			wg.Add(1)
			go func(ctx context.Context) {
				defer wg.Done()

				req1 := pimage.ImageCensorReq{}
				req1.Data.IMG = uri
				if scenem != nil {
					req1.Params = scenem[TERROR]
				}
				resp1, err1 := s.TerrorCensor(ctx, req1)
				if err1 != nil {
					lock.Lock()
					defer lock.Unlock()
					if err == nil {
						err = err1
					}
					return
				}
				lock.Lock()
				defer lock.Unlock()

				resp.Scenes[TERROR] = resp1
				resp.Suggestion = resp.Suggestion.Update(resp1.Suggestion)

			}(util.SpawnContext(ctx))
		case POLITICIAN:
			wg.Add(1)
			go func(ctx context.Context) {
				defer wg.Done()

				req1 := pimage.ImageCensorReq{}
				req1.Data.IMG = uri
				resp1, err1 := s.PoliticianCensor(ctx, req1)
				if err1 != nil {
					lock.Lock()
					defer lock.Unlock()
					if err == nil {
						err = err1
					}
					return
				}
				lock.Lock()
				defer lock.Unlock()

				resp.Scenes[POLITICIAN] = resp1
				resp.Suggestion = resp.Suggestion.Update(resp1.Suggestion)

			}(util.SpawnContext(ctx))
		case ADS:
			wg.Add(1)
			go func(ctx context.Context) {
				defer wg.Done()

				req1 := pimage.ImageCensorReq{}
				req1.Data.IMG = uri
				if scenem != nil {
					req1.Params = scenem[ADS]
				}
				resp1, err1 := s.AdsCensor(ctx, req1)
				if err1 != nil {
					lock.Lock()
					defer lock.Unlock()
					if err == nil {
						err = err1
					}
					return
				}
				lock.Lock()
				defer lock.Unlock()

				resp.Scenes[ADS] = resp1
				resp.Suggestion = resp.Suggestion.Update(resp1.Suggestion)

			}(util.SpawnContext(ctx))
		}
	}

	wg.Wait()
	return resp, err
}
