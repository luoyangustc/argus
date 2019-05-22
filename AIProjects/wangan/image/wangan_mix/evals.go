package wangan_mix

import (
	"context"

	"github.com/go-kit/kit/endpoint"
	"qiniu.com/argus/service/service/image"
)

type EvalWanganMixReq struct {
	Data struct {
		IMG image.Image
	} `json:"data"`
}

type EvalWanganMixResp struct {
	Code    int                 `json:"code"`
	Message string              `json:"message"`
	Result  EvalWanganMixResult `json:"result"`
}

type EvalWanganMixResult struct {
	Classify  EvalWanganMixClassify    `json:"classify"`
	Detection []EvalWanganMixDetection `json:"detection"`
}

type EvalWanganMixClassify struct {
	Confidences []struct {
		Class string  `json:"class"`
		Index int     `json:"index"`
		Score float32 `json:"score"`
	} `json:"confidences"`
}

type EvalWanganMixDetection struct {
	Class string    `json:"class"`
	Index int       `json:"index"`
	Score float32   `json:"score"`
	Pts   [4][2]int `json:"pts"`
}

type EvalWanganMixService interface {
	EvalWanganMix(context.Context, EvalWanganMixReq) (EvalWanganMixResp, error)
}

type EvalWanganMixEndPoints struct {
	EvalWanganMixEP endpoint.Endpoint
}

func (ends EvalWanganMixEndPoints) EvalWanganMix(ctx context.Context, req EvalWanganMixReq) (EvalWanganMixResp, error) {
	resp, err := ends.EvalWanganMixEP(ctx, req)
	if err != nil {
		return EvalWanganMixResp{}, err
	}
	return resp.(EvalWanganMixResp), nil
}
