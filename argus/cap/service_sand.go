package cap

import (
	"context"
	"strconv"

	"github.com/qiniu/http/restrpc.v1"
	xlog "github.com/qiniu/xlog.v1"
	"qiniu.com/argus/cap/model"
	"qiniu.com/argus/cap/sand"
)

////////////////////////////////////////////////////////////////////////////////

type ISandService interface {
	GetSands(context.Context, *model.GetSandsReq, *restrpc.Env) (*model.GetSandsResp, error)
	PostAddSandFiles(ctx context.Context, req *model.AddSandFilesRequest, env *restrpc.Env) error
	PostAddSand(ctx context.Context, req *model.AddSandRequest, env *restrpc.Env) error
}

// NewAuditService NewAuditService
func NewSandService(sandMixer sand.ISandMixer) (ISandService, error) {
	return &_SandService{
		ISandMixer: sandMixer,
	}, nil
}

////////////////////////////////////////////////////////////////////////////////

type _SandService struct {
	sand.ISandMixer
}

func (s *_SandService) GetSands(
	ctx context.Context,
	req *model.GetSandsReq,
	env *restrpc.Env,
) (*model.GetSandsResp, error) {
	err := env.Req.ParseForm()
	if err != nil {
		return nil, err
	}
	labelType := env.Req.Form.Get("label_type")
	num, err := strconv.Atoi(env.Req.Form.Get("num"))
	if err != nil {
		return nil, err
	}
	ret := s.ISandMixer.QuerySandsByType(ctx, num, labelType)
	resp := model.GetSandsResp{
		Data: ret,
	}
	return &resp, nil
}

func (s *_SandService) PostAddSandFiles(ctx context.Context, req *model.AddSandFilesRequest, env *restrpc.Env,
) error {

	xl := xlog.FromContextSafe(ctx)
	xl.Infof("PostAddSandFile req, %+v", req)

	for _, file := range req.Files {
		err := s.ISandMixer.AddSandFileByURL(file)
		if err != nil {
			xl.Errorf("AddSandFileByURL error:%s", err)
			return err
		}
	}
	return nil
}

func (s *_SandService) PostAddSand(ctx context.Context, req *model.AddSandRequest, env *restrpc.Env,
) error {

	xl := xlog.FromContextSafe(ctx)
	xl.Infof("PostAddSand req, %+v", req)

	var sandArr []model.TaskModel
	for _, task := range req.Tasks {
		sand := model.TaskModel{
			URI:    task.URI,
			Labels: task.Label,
		}
		sandArr = append(sandArr, sand)
	}

	return s.ISandMixer.AddSand(ctx, sandArr...)
}
