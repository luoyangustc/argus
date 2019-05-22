package police

import (
	"context"

	"github.com/go-kit/kit/endpoint"
	xlog "github.com/qiniu/xlog.v1"
	. "qiniu.com/argus/service/service"

	pimage "qiniu.com/argus/service/service/image"
)

type PoliceReq struct {
	Data struct {
		IMG pimage.Image
		// URI string `json:"uri"`
	} `json:"data"`
	Params struct {
		Detail bool `json:"detail"`
	} `json:"params"`
}

type PoliceDetection struct {
	Class string   `json:"class"`
	Pts   [][2]int `json:"pts"`
	Score float32  `json:"score"`
}

type PoliceResult struct {
	Label      int               `json:"label"`
	Score      float32           `json:"score"`
	Detections []PoliceDetection `json:"detections,omitempty"`
}

type PoliceResp struct {
	Code    int          `json:"code"`
	Message string       `json:"message"`
	Result  PoliceResult `json:"result"`
}

type PoliceService interface {
	Police(context.Context, PoliceReq) (PoliceResp, error)
}

var _ PoliceService = &PoliceEndPoints{}

type PoliceEndPoints struct {
	PoliceEP endpoint.Endpoint
}

func (eps PoliceEndPoints) Police(ctx context.Context, req PoliceReq) (PoliceResp, error) {
	response, err := eps.PoliceEP(ctx, req)
	if err != nil {
		return PoliceResp{}, err
	}
	return response.(PoliceResp), nil
}

type Config struct{}

var (
	DEFAULT = Config{}
)

type policeService struct {
	Config
	EvalPoliceService
}

func NewPoliceService(
	conf Config,
	eps EvalPoliceService,
) (PoliceService, error) {
	return &policeService{
		Config:            conf,
		EvalPoliceService: eps,
	}, nil
}

func (s *policeService) Police(ctx context.Context, req PoliceReq) (PoliceResp, error) {
	var (
		req0       EvalPoliceReq
		xl         = xlog.FromContextSafe(ctx)
		detections []PoliceDetection
		score      float32 = -1.0
	)
	req0.Data.IMG = req.Data.IMG
	resp, e0 := s.EvalPolice(ctx, req0)
	if e0 != nil {
		err := ErrInternal(e0.Error())
		xl.Errorf("call eval police failed. error: %v", err)
		return PoliceResp{}, err
	}
	for _, detect := range resp.Result.Detections {
		detections = append(detections, PoliceDetection{
			Class: detect.Class,
			Score: detect.Score,
			Pts:   detect.Pts,
		})
		if detect.Score > score {
			score = detect.Score
		}
	}
	var result PoliceResult
	if score > 0 {
		result.Label = 1
		result.Score = score
		if req.Params.Detail {
			result.Detections = detections
		}
	}
	return PoliceResp{
		Code:    resp.Code,
		Message: resp.Message,
		Result:  result,
	}, nil
}
