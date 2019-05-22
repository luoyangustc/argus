package objectdetect

import (
	"context"
	"strings"

	"github.com/go-kit/kit/endpoint"
	xlog "github.com/qiniu/xlog.v1"
	. "qiniu.com/argus/service/service"
)

type DetectionReq EvalObjectDetectReq
type DetectionResp EvalObjectDetectResp

type ObjectDetectService interface {
	ObjectDetect(context.Context, DetectionReq) (DetectionResp, error)
}

var _ ObjectDetectService = ObjectDetectEndpoints{}

type ObjectDetectEndpoints struct {
	ObjectDetectEP endpoint.Endpoint
}

func (ends ObjectDetectEndpoints) ObjectDetect(ctx context.Context, req DetectionReq) (
	DetectionResp, error) {
	response, err := ends.ObjectDetectEP(ctx, req)
	if err != nil {
		return DetectionResp{}, err
	}
	resp := response.(DetectionResp)
	return resp, err
}

type objectDetectService struct {
	eOD EvalObjectDetectService
}

func NewObjectDetectService(eod EvalObjectDetectService) (ObjectDetectService, error) {
	return objectDetectService{eOD: eod}, nil
}

//------------------------------------------------------------//

func (s objectDetectService) ObjectDetect(ctx context.Context, args DetectionReq) (ret DetectionResp, err error) {

	// ctex, xl = ctxAndLog(ctx, env.W, env.Req)
	var xl = xlog.FromContextSafe(ctx)

	if strings.TrimSpace(string(args.Data.IMG.URI)) == "" {
		xl.Error("empty data.uri")
		return ret, ErrArgs("empty data.uri")
	}

	var dtReq EvalObjectDetectReq
	dtReq.Data.IMG.URI = args.Data.IMG.URI
	resp, err := s.eOD.EvalObjectDetect(ctx, dtReq)
	if err != nil {
		xl.Errorf("call object detection error:%v", err)
		return ret, ErrInternal(err.Error())
	}
	ret = DetectionResp(resp)

	return ret, nil
}
