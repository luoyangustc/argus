package cap

import (
	"context"
	"encoding/json"
	"errors"

	"qiniu.com/argus/argus/com/util"
	"qiniu.com/argus/cap/dao"
	"qiniu.com/argus/cap/enums"
	"qiniu.com/argus/cap/model"
	"qiniu.com/argus/cap/task"

	restrpc "github.com/qiniu/http/restrpc.v1"
	xlog "github.com/qiniu/xlog.v1"
)

// IResultService
type IResultService interface {
	GetJob_Results(
		context.Context,
		*model.GetResultReq,
		*restrpc.Env,
	) error
	PostJob_CheckResult(
		context.Context,
		*model.JobCheckResultReq,
		*restrpc.Env,
	) (*model.JobCheckResultResp, error)
}

var _ IResultService = _ResultService{}

type _ResultService struct {
	task.IBatchResultHandler
	dao.IJobDAO
	dao.ITaskDAO
}

func NewResultService(bResultHandler task.IBatchResultHandler, jDao dao.IJobDAO, tDao dao.ITaskDAO) _ResultService {
	return _ResultService{
		IBatchResultHandler: bResultHandler,
		IJobDAO:             jDao,
		ITaskDAO:            tDao,
	}
}

func (s _ResultService) GetJob_Results(
	ctx context.Context,
	req *model.GetResultReq,
	env *restrpc.Env,
) error {
	if req == nil || len(req.CmdArgs) <= 0 {
		return errors.New("invalid CmdArgs")
	}

	ctx, xl := util.CtxAndLog(ctx, env.W, env.Req)
	xl.Infof("GetJobsResult, %#v", req)

	var (
		jid    = req.CmdArgs[0]
		writer = func(task model.TaskModel) error {

			env.W.Write([]byte(task.TaskID))
			env.W.Write([]byte("\t"))
			env.W.Write([]byte(task.URI))
			env.W.Write([]byte("\t"))
			bs, _ := json.Marshal(task.Result)
			env.W.Write(bs)
			env.W.Write([]byte("\n"))
			return nil
		}
	)

	jobInMgo, err := s.IJobDAO.QueryByID(ctx, jid)
	if err != nil {
		xl.Errorf("s.IJobDAO.QueryByID", err.Error())
		return err
	}
	if jobInMgo.JobType == string(enums.REALTIME) {
		//TODO：增量图片的处理
	} else {
		_, err = s.all(ctx, jobInMgo.LabelMode, jid, req.Marker, req.Limit, 100, writer)
	}

	return err

}

func (s _ResultService) PostJob_CheckResult(
	ctx context.Context,
	req *model.JobCheckResultReq,
	env *restrpc.Env,
) (*model.JobCheckResultResp, error) {
	if req == nil || len(req.CmdArgs) <= 0 {
		return nil, errors.New("invalid CmdArgs")
	}

	var (
		xl   = xlog.FromContextSafe(ctx)
		resp = model.JobCheckResultResp{}
	)
	bFinish, err := s.IBatchResultHandler.CheckTasksDone(ctx, req.CmdArgs[0])
	if err != nil {
		xl.Errorf("s.IBatchResultHandler.CheckTasksDone error: %#v", err.Error())
		resp.Finish = false
		return &resp, err
	}

	xl.Infof("the job %#v status is %#v", req.CmdArgs[0], bFinish)
	resp.Finish = bFinish
	return &resp, nil
}

//===========================================================================================
func (s _ResultService) all(ctx context.Context, mode, jid, marker string, limit, batchSize int,
	writer func(model.TaskModel) error,
) (string, error) {
	var (
		n         = 0
		err       error
		retMarker = marker
		xl        = xlog.FromContextSafe(ctx)
	)

	xl.Infof("cap limit: %#v", limit)

	for {
		if limit > 0 && batchSize > limit-n {
			batchSize = limit - n
		}
		if batchSize <= 0 {
			break
		}

		var tasks []dao.TaskInMgo
		retMarker, tasks, err = s.ITaskDAO.QueryResults(ctx, mode, jid, retMarker, batchSize)
		if err != nil {
			xl.Infof("s.ITaskDAO.QueryResults error: %#v", err.Error())
			return "", err
		}
		for _, v := range tasks {
			task := model.FromTaskInMgo(&v)
			if err = writer(*task); err != nil {
				xl.Infof("writer error: %#v", err.Error())
				return "", err
			}
		}
		if len(tasks) < batchSize {
			break
		}
		n += len(tasks)
	}

	return retMarker, nil
}
