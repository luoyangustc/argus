package tuso_job

import (
	"context"
	"encoding/json"

	job "qiniu.com/argus/bjob/proto"
	"qiniu.com/argus/tuso/proto"
)

type Processor interface {
	Process(context.Context, interface{}) error
}

var _ job.JobCreator = ProcessNode{}

type ProcessNode struct {
}

func (node ProcessNode) NewMaster(ctx context.Context, reqBody []byte, env job.Env) (job.JobMaster, error) {
	var hubs = []string{}
	_ = json.Unmarshal(reqBody, &hubs)
	return &ProcessMaster{hubs: hubs, index: 0}, nil
}

//----------------------------------------------------------------------------//

type ProcessMaster struct {
	hubs  []string
	index int
}

func (m *ProcessMaster) NextTask(ctx context.Context) ([]byte, string, bool) {
	if m.index >= len(m.hubs) {
		return nil, "", false
	}
	var hub = m.hubs[m.index]
	m.index += 1
	bs, _ := json.Marshal(proto.ProcessTask{Hub: hub})
	return bs, "", true
}
func (m ProcessMaster) Error(ctx context.Context) error { return nil }
func (m ProcessMaster) Stop(ctx context.Context)        {}
func (m ProcessMaster) AppendResult(ctx context.Context, result job.TaskResult) error {
	return nil
}
func (m ProcessMaster) Result(ctx context.Context) ([]byte, error) { return nil, nil }

//----------------------------------------------------------------------------//

type ProcessWorker struct {
	proto.HubMgr
	proto.MetaMgr
	proto.LogMgr
	proto.FeatureFileMgr
}

func (w ProcessWorker) Do(ctx context.Context, task job.Task) ([]byte, error) {
	var _task proto.ProcessTask
	_ = json.Unmarshal(task.Value(ctx), &_task)

	// TODO query logs -> write blocks

	return nil, nil
}
