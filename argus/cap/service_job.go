package cap

import (
	"context"
	"errors"
	"fmt"
	"strconv"
	"time"

	restrpc "github.com/qiniu/http/restrpc.v1"
	xlog "github.com/qiniu/xlog.v1"
	"qiniu.com/argus/cap/enums"
	"qiniu.com/argus/cap/job"
	"qiniu.com/argus/cap/model"
)

// IJobService
type IJobService interface {
	GetJob_(context.Context, *model.JobQueryReq, *restrpc.Env,
	) (*model.JobQueryResp, error)

	PostJob(context.Context, *model.JobCreateReq, *restrpc.Env,
	) (*model.JobCreateResp, error)

	PostJob_Tasks(context.Context, *model.JobTaskReq, *restrpc.Env,
	) error

	GetJob_Task_(context.Context, *model.JobQueryReq, *restrpc.Env,
	) (*model.TaskModel, error)
}

////////////////////////////////////////////////////////////////

var _ IJobService = &_JobService{}

type _JobService struct {
	job.IJobHandler
}

func NewJobService(jobHandler *job.IJobHandler,
) IJobService {
	return &_JobService{
		IJobHandler: *jobHandler,
	}
}

func (serv *_JobService) GetJob_(
	ctx context.Context,
	req *model.JobQueryReq,
	env *restrpc.Env,
) (*model.JobQueryResp, error) {
	if req == nil || len(req.CmdArgs) <= 0 {
		return nil, errors.New("invalid CmdArgs")
	}

	return serv.IJobHandler.QueryJobById(ctx, req.CmdArgs[0])
}

func (serv *_JobService) PostJob(
	ctx context.Context,
	req *model.JobCreateReq,
	env *restrpc.Env,
) (*model.JobCreateResp, error) {
	var (
		xl = xlog.FromContextSafe(ctx)
	)

	xl.Infof("PostJob req, %+v", req)

	// 补全jobID
	if req.JobID == "" {
		req.JobID = strconv.FormatInt(time.Now().Unix(), 10)
	}

	// 幂等检查
	job, err := serv.IJobHandler.QueryJobById(ctx, req.JobID)
	if err == nil {
		xl.Infof("job has existed, %+v", job.JobID)
		if job.Status == enums.JobBegin {
			// 已创建，状态正常
			return &model.JobCreateResp{
				JobID: req.JobID,
			}, nil
		} else {
			// 已创建，状态不可用
			return nil, fmt.Errorf("job %s already exist", req.JobID)
		}
	}

	if !req.MimeType.IsValid() {
		return nil, fmt.Errorf("req.MimeType invalid, expected is %#v, %#v or %#v, actual is %#v", enums.MimeTypeImage, enums.MimeTypeVideo, enums.MimeTypeLive, req.MimeType)
	}
	if !req.JobType.IsValid() {
		return nil, fmt.Errorf("req.JobType invalid, expected is %#v, %#v, actual is %#v", enums.REALTIME, enums.BATCH, req.JobType)
	}

	//TODO: 增量的处理方式？
	if req.JobType == enums.REALTIME {
		// err := serv.JobManager.NewStreamJob(ctx, req.JobID, req.LabelMode, req.NotifyURL, 1)
		// if err != nil {
		// 	xl.Infof("NewStreamJob err, %v", err)
		// 	return nil, err
		// }
	} else {
		job := model.JobModel{
			JobID:     req.JobID,
			JobType:   enums.JobType(req.JobType),
			LabelMode: req.LabelMode,
			MimeType:  enums.MimeType(req.MimeType),
		}
		err := serv.IJobHandler.NewJob(ctx, &job)
		if err != nil {
			xl.Infof("NewJob err, %#v", err)
			return nil, err
		}
	}

	xl.Infof("NewJob done: %s", req.JobID)
	return &model.JobCreateResp{
		JobID: req.JobID,
	}, nil
}

func (serv *_JobService) PostJob_Tasks(
	ctx context.Context,
	req *model.JobTaskReq,
	env *restrpc.Env,
) error {
	if req == nil || len(req.CmdArgs) <= 0 {
		return errors.New("invalid CmdArgs")
	}
	var (
		xl    = xlog.FromContextSafe(ctx)
		tasks = make([]model.TaskModel, 0)
	)

	xl.Infof("PostJob_Tasks req, %+v", req)

	for _, v := range req.Tasks {
		var task model.TaskModel
		task.JobID = req.CmdArgs[0]
		task.TaskID = v.ID
		task.URI = v.URI
		task.Labels = v.Labels
		tasks = append(tasks, task)
	}

	if len(tasks) <= 0 {
		xl.Infof("tasks empty err, %s", req.CmdArgs[0])
		return errors.New("tasks empty")
	}

	xl.Infof("tasks empty err, %s", req.CmdArgs[0])
	err := serv.IJobHandler.PushTasks(ctx, req.CmdArgs[0], tasks)
	if err != nil {
		xl.Infof("Push err, %s", req.CmdArgs[0])
		return err
	}

	xl.Infof("PostTasks ok, %s", req.CmdArgs[0])
	return nil
}

func (serv *_JobService) GetJob_Task_(ctx context.Context,
	req *model.JobQueryReq,
	env *restrpc.Env,
) (*model.TaskModel, error) {
	if req == nil || len(req.CmdArgs) <= 1 {
		return nil, errors.New("invalid CmdArgs")
	}

	return serv.IJobHandler.QueryTaskbById(ctx, req.CmdArgs[0], req.CmdArgs[1])
}
