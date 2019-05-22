package cap

import (
	"context"

	// "qiniu.com/argus/cap/dao"

	"github.com/qiniu/http/restrpc.v1"
	xlog "github.com/qiniu/xlog.v1"
	"qiniu.com/argus/argus/com/util"
	"qiniu.com/argus/cap/auditor"
	"qiniu.com/argus/cap/model"
)

////////////////////////////////////////////////////////////////////////////////

type IAuditService interface {
	GetAuditorAttr_(
		context.Context,
		*model.GetAuditorAttrReq,
		*restrpc.Env,
	) (*model.GetAuditorAttrResp, error)

	GetRealtimeTask_(
		context.Context,
		*model.GetRealtimeTaskReq,
		*restrpc.Env,
	) (*model.GetRealtimeTaskResp, error)

	PostCancelRealtimeTask(
		context.Context,
		*model.PostCancelRealtimeTaskReq,
		*restrpc.Env,
	) (*model.PostCancelRealtimeTaskResp, error)

	PostResult(
		context.Context,
		*model.PostResultReq,
		*restrpc.Env,
	) (*model.PostResultResp, error)
}

// NewAuditService NewAuditService
func NewAuditService(as auditor.IAuditor) (IAuditService, error) {
	return &_AuditService{
		auditor: as,
	}, nil
}

////////////////////////////////////////////////////////////////////////////////

type _AuditService struct {
	auditor auditor.IAuditor
}

func (adServ *_AuditService) GetAuditorAttr_(
	ctx context.Context,
	req *model.GetAuditorAttrReq,
	env *restrpc.Env,
) (*model.GetAuditorAttrResp, error) {
	return adServ.auditor.GetAuditorAttr(ctx, req.CmdArgs[0])
}

func (adServ *_AuditService) GetRealtimeTask_(
	ctx context.Context,
	req *model.GetRealtimeTaskReq,
	env *restrpc.Env,
) (*model.GetRealtimeTaskResp, error) {
	var (
		xl = xlog.FromContextSafe(ctx)
	)
	xl.Infof("GetRealtimeTask_ req: %#v", req.CmdArgs[0])
	ctx, _ = util.CtxAndLog(ctx, env.W, env.Req)

	tasks, err := adServ.auditor.FetchTasks(ctx, req.CmdArgs[0])
	if err != nil || len(tasks.IndexData) == 0 {
		return nil, err
	}
	return &tasks, err
}

func (adServ *_AuditService) PostCancelRealtimeTask(
	ctx context.Context,
	req *model.PostCancelRealtimeTaskReq,
	env *restrpc.Env,
) (*model.PostCancelRealtimeTaskResp, error) {
	xl := xlog.FromContextSafe(ctx)
	xl.Infof("PostCancelRealtimeTask for aid: %v", req.AuditorID)
	ctx, _ = util.CtxAndLog(ctx, env.W, env.Req)

	err := adServ.auditor.CancelTasks(ctx, req.AuditorID, req.TaskIds, req.PID)

	if err != nil {
		xl.Errorf("adServ.auditor.CancelTasks error: %#v", err.Error())
		return &model.PostCancelRealtimeTaskResp{Code: 555, Msg: err.Error()}, err
	}

	return &model.PostCancelRealtimeTaskResp{Code: 200, Msg: ""}, nil
}

func (adServ *_AuditService) PostResult(
	ctx context.Context,
	req *model.PostResultReq,
	env *restrpc.Env,
) (*model.PostResultResp, error) {
	xl := xlog.FromContextSafe(ctx)
	xl.Infof("PostResult: %v", req.AuditorID)
	ctx, _ = util.CtxAndLog(ctx, env.W, env.Req)

	if req.Success {
		err := adServ.auditor.SaveTasks(ctx, req.AuditorID, req.Result, req.PID)
		if err != nil {
			xl.Errorf("adServ.auditor.SaveTasks error: %#v", err.Error())
			return &model.PostResultResp{Code: 555, Msg: err.Error()}, err
		}
		return &model.PostResultResp{Code: 200, Msg: ""}, nil
	}
	return &model.PostResultResp{Code: 400, Msg: "illegal request"}, nil
}
