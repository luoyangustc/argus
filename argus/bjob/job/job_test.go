package job

import (
	"context"
	"encoding/json"
	"testing"

	. "qiniu.com/argus/bjob/proto"
)

func TestJobMock(t *testing.T) {

	var creator MockJobCreator
	var worker MockTaskWorker

	var (
		master JobMaster
	)

	{
		var (
			ctx = context.Background()
			req = make([]byte, 0) // MOCK
		)
		master, _ = creator.NewMaster(ctx, req, Env{})
	}

	{
		var ctx = context.Background()
		for {
			taskBody, _, notEnd := master.NextTask(ctx)
			if !notEnd {
				break
			}

			task := MockTask{}
			task.Value_, _ = json.Marshal(taskBody)

			resultBody, _ := worker.Do(ctx, task)

			taskResult := MockTaskResult{MockTask: task}
			taskResult.Result, _ = json.Marshal(resultBody)

			_ = master.AppendResult(ctx, taskResult)
		}
		if err := master.Error(ctx); err != nil {
			return
		}

		_, _ = master.Result(ctx)
	}
}
