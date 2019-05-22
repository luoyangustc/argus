package task

import (
	"context"

	xlog "github.com/qiniu/xlog.v1"
	"qiniu.com/argus/cap/dao"
)

type IBatchResultHandler interface {
	CheckTasksDone(context.Context, string) (bool, error)
}

func NewBatchResult(jDao dao.IJobDAO, taskDao dao.ITaskDAO) IBatchResultHandler {
	return _BatchResultHandler{
		IJobDAO:  jDao,
		ITaskDAO: taskDao,
	}
}

var _ IBatchResultHandler = _BatchResultHandler{}

type _BatchResultHandler struct {
	dao.IJobDAO
	dao.ITaskDAO
}

func (b _BatchResultHandler) CheckTasksDone(ctx context.Context, jId string) (bool, error) {
	var (
		xl = xlog.FromContextSafe(ctx)
	)

	jInMgo, err := b.IJobDAO.QueryByID(ctx, jId)
	if err != nil {
		xl.Errorf(" b.IJobDAO.QueryByID error: %#v", err.Error())
		return false, err
	}

	tNotDonCount, err := b.ITaskDAO.QueryNotDoneCountByJid(ctx, jInMgo.LabelMode, jId)
	if err != nil {
		xl.Errorf("b.ITaskDAO.QueryByJidAndStatus error: %#v", err.Error())
		return false, err
	}
	return tNotDonCount == 0, nil
}
