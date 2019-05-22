package graffiti

import (
	"context"

	"github.com/go-kit/kit/endpoint"
	xlog "github.com/qiniu/xlog.v1"
	. "qiniu.com/argus/service/service"
	pimage "qiniu.com/argus/service/service/image"
)

type GraffitiReq struct {
	Data struct {
		IMG pimage.Image
	} `json:"data"`
}

type GraffitiDetection struct {
	Index       int                `json:"index"`
	Class       string             `json:"class"`
	BoundingBox pimage.BoundingBox `json:"bounding_box"`
}

type GraffitiResp struct {
	Code    int    `json:"code"`
	Message string `json:"message"`
	Result  struct {
		Detections []GraffitiDetection `json:"detections"`
	} `json:"result"`
}

type GraffitiService interface {
	Graffiti(context.Context, GraffitiReq) (GraffitiResp, error)
}

var _ GraffitiService = GraffitiEndPoints{}

type GraffitiEndPoints struct {
	GraffitiEP endpoint.Endpoint
}

func (ends GraffitiEndPoints) Graffiti(ctx context.Context, req GraffitiReq) (GraffitiResp, error) {
	response, err := ends.GraffitiEP(ctx, req)
	if err != nil {
		return GraffitiResp{}, err
	}
	return response.(GraffitiResp), nil
}

type Config struct {
}

var (
	DEFAULT = Config{}
)

type graffitiService struct {
	Config
	eGraffiti EvalGraffitiDetectService
}

func NewGraffitiService(cfg Config, gs EvalGraffitiDetectService) (GraffitiService, error) {
	return graffitiService{Config: cfg, eGraffiti: gs}, nil
}

func (s graffitiService) Graffiti(ctx context.Context, req GraffitiReq) (GraffitiResp, error) {
	var (
		xl    = xlog.FromContextSafe(ctx)
		gdReq GraffitiDetectReq
		ret   GraffitiResp
	)

	gdReq.Data.IMG = req.Data.IMG
	gdResp, err := s.eGraffiti.EvalGraffitiDetect(ctx, gdReq)
	if err != nil {
		xl.Errorf("call graffiti detect error: %s", err.Error())
		return GraffitiResp{}, ErrInternal(err.Error())
	}

	if gdResp.Code != 0 && gdResp.Code/100 != 2 {
		xl.Errorf("call graffiti detect failed, resp: %v", gdResp)
		return GraffitiResp{}, ErrInternal(gdResp.Message)
	}

	for _, dt := range gdResp.Result {
		ret.Result.Detections = append(ret.Result.Detections, GraffitiDetection{
			Index: dt.Index,
			Class: dt.Class,
			BoundingBox: pimage.BoundingBox{
				Pts:   dt.Pts,
				Score: dt.Score,
			},
		})
	}

	if len(ret.Result.Detections) == 0 {
		ret.Message = "No graffiti found"
	}

	return ret, nil
}
