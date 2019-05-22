package job

import (
	"context"

	"github.com/prometheus/client_golang/prometheus"
	xlog "github.com/qiniu/xlog.v1"

	"qiniu.com/argus/cap/dao"
	"qiniu.com/argus/cap/enums"
	"qiniu.com/argus/cap/model"
)

// JobManager interface
type IJobHandler interface {
	QueryJobById(context.Context, string) (*model.JobQueryResp, error)
	NewJob(context.Context, *model.JobModel) error

	QueryTaskbById(context.Context, string, string) (*model.TaskModel, error)
	PushTasks(context.Context, string, []model.TaskModel) error

	// Pop(context.Context, string, string, int) ([]proto.TaskInfo, []string, error)
	// Confirm(ctx context.Context, msgid string) error
}

type _JobHandler struct {
	dao.IJobDAO
	dao.ITaskDAO
}

func NewJobHandler(
	jobDao dao.IJobDAO,
	taskDao dao.ITaskDAO,
) IJobHandler {
	return _JobHandler{
		IJobDAO:  jobDao,
		ITaskDAO: taskDao,
	}
}

func (jh _JobHandler) QueryJobById(ctx context.Context, jobId string) (*model.JobQueryResp, error) {
	jobMgo, err := jh.IJobDAO.QueryByID(ctx, jobId)
	if err != nil {
		return nil, err
	}

	resp := model.JobQueryResp{}

	resp.JobModel = *model.FromJobInMgo(jobMgo)

	return &resp, nil
}

func (jh _JobHandler) NewJob(ctx context.Context, jobModel *model.JobModel) error {
	return jh.IJobDAO.Insert(ctx, *model.ToJobInMgo(jobModel))
}

var _TaskPushGaugeVec = func() *prometheus.GaugeVec {
	vec := prometheus.NewGaugeVec(
		prometheus.GaugeOpts{
			Namespace: "argus",
			Subsystem: "cap",
			Name:      "task_push",
			Help:      "task push",
		},
		[]string{"jid", "type", "mode"},
	)
	prometheus.MustRegister(vec)
	return vec
}()

func (jh _JobHandler) PushTasks(ctx context.Context, jobId string, tasks []model.TaskModel) error {
	var (
		xl = xlog.FromContextSafe(ctx)
	)
	jobMgo, err := jh.IJobDAO.QueryByID(ctx, jobId)
	if err != nil {
		return err
	}

	if jobMgo.JobType == string(enums.REALTIME) {
		//TODO:保存实时的任务
		xl.Infof("the job is realtime")
		return nil
	} else {
		//Batch Tasks
		tasksInMgo := make([]dao.TaskInMgo, 0)
		for _, v := range tasks {
			tInMgo := model.ToTaskInMgo(&v)
			tasksInMgo = append(tasksInMgo, *tInMgo)
		}
		err := jh.ITaskDAO.Insert(ctx, jobMgo.LabelMode, tasksInMgo...)
		if err != nil {
			xl.Errorf("jh.ITaskDAO.Insert error: %#v", err.Error())
			return err
		}
	}

	_TaskPushGaugeVec.
		WithLabelValues(jobMgo.JobID, jobMgo.JobType, jobMgo.LabelMode).
		Add(float64(len(tasks)))
	return nil
}

func (jh _JobHandler) QueryTaskbById(ctx context.Context, jobId, taskId string) (*model.TaskModel, error) {
	var (
		xl = xlog.FromContextSafe(ctx)
	)
	jobMgo, err := jh.IJobDAO.QueryByID(ctx, jobId)
	if err != nil {
		return nil, err
	}

	taskMgo, err := jh.ITaskDAO.QueryByID(ctx, jobMgo.LabelMode, taskId)
	if err != nil {
		return nil, err
	}

	taskModel := model.FromTaskInMgo(taskMgo)
	xl.Infof("taskModel: %#v", taskModel.Labels)
	xl.Infof("taskModel Result: %#v", taskModel.Result)

	return model.FromTaskInMgo(taskMgo), nil
}
