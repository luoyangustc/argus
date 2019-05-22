package police

import (
	"context"

	"github.com/go-kit/kit/endpoint"
	"qiniu.com/argus/service/service/biz"
	pimage "qiniu.com/argus/service/service/image"
)

type EvalPoliceReq struct {
	Data struct {
		IMG pimage.Image
		// URI string `json:"uri"`
	} `json:"data"`
}

type EvalPoliceDetection struct {
	Index int      `json:"index"`
	Class string   `json:"class"`
	Score float32  `json:"score"`
	Pts   [][2]int `json:"pts"`
}

type EvalPoliceResult struct {
	Detections []EvalPoliceDetection `json:"detections"`
}

type EvalPoliceResp struct {
	Code    int              `json:"code"`
	Message string           `json:"message"`
	Result  EvalPoliceResult `json:"result"`
}

type EvalPoliceService interface {
	EvalPolice(context.Context, EvalPoliceReq) (EvalPoliceResp, error)
}

type EvalPoliceEndPoints struct {
	EvalPoliceEP endpoint.Endpoint
}

func (eps EvalPoliceEndPoints) EvalPolice(ctx context.Context, req EvalPoliceReq) (EvalPoliceResp, error) {
	resp, err := eps.EvalPoliceEP(ctx, req)
	if err != nil {
		return EvalPoliceResp{}, err
	}
	return resp.(EvalPoliceResp), nil
}

var EVAL_POLICE_CONFIG = biz.EvalModelConfig{
	Image: "hub2.qiniu.com/1381102897/ava-eval-police-detect-refinedet:201901251630",
	Model: "ava-police/police-detect-refinedet-20180504-V2.tar",
	Args: &biz.ModelConfigArgs{
		BatchSize: 16,
		CustomValues: map[string]interface{}{
			"thresholds": []float32{0, 0.6, 0.6, 0.6, 0.6},
		},
	},
	Type: biz.EvalRunTypeServing,
}
