package scene

import (
	"context"
	"strings"

	"github.com/go-kit/kit/endpoint"
	xlog "github.com/qiniu/xlog.v1"
	. "qiniu.com/argus/service/service"
)

type SceneReq EvalSceneReq
type SceneResp EvalSceneResp

type SceneService interface {
	Scene(ctx context.Context, req SceneReq) (SceneResp, error)
}

var _ SceneService = SceneEndpoints{}

type SceneEndpoints struct {
	SceneEP endpoint.Endpoint
}

func (ends SceneEndpoints) Scene(ctx context.Context, req SceneReq) (SceneResp, error) {
	response, err := ends.SceneEP(ctx, req)
	if err != nil {
		return SceneResp{}, err
	}
	resp := response.(SceneResp)
	return resp, nil
}

type sceneService struct {
	EvalSceneService
}

func NewSceneService(ess EvalSceneService) (SceneService, error) {
	return sceneService{EvalSceneService: ess}, nil
}

func (s sceneService) Scene(ctx context.Context, req SceneReq) (ret SceneResp, err error) {

	var (
		xl = xlog.FromContextSafe(ctx)
	)

	if strings.TrimSpace(string(req.Data.IMG.URI)) == "" {
		xl.Error("empty data.uri")
		return ret, ErrArgs("empty data.uri")
	}

	var dtReq EvalSceneReq
	dtReq.Data.IMG.URI = req.Data.IMG.URI
	resp, err := s.EvalScene(ctx, dtReq)
	if err != nil {
		xl.Errorf("call scene error:%v", err)
		return ret, ErrInternal(err.Error())
	}
	ret = SceneResp(resp)

	return ret, nil
}
