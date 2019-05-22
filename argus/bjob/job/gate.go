package job

import (
	"context"
	"encoding/json"

	"github.com/qiniu/xlog.v1"
	"qiniu.com/auth/authstub.v1"

	. "qiniu.com/argus/bjob/proto"
)

type Gate struct {
	MQs
}

func NewGate(mqs MQs) Gate {
	return Gate{MQs: mqs}
}

// Get /query/<cmd>/<id>
func (g Gate) GetQuery__(
	ctx context.Context,
	req *struct {
		CmdArgs []string
	},
	env *authstub.Env,
) (struct {
	Status JobStatus        `json:"status"`
	Error  string           `json:"error,omitempty"`
	Result *json.RawMessage `json:"result,omitempty"`
}, error) {

	xl, ok := xlog.FromContext(ctx)
	if !ok {
		xl = xlog.NewWithReq(env.Req)
		ctx = xlog.NewContext(ctx, xl)
	}
	_ = xl

	var (
		cmd = req.CmdArgs[0]
		id  = req.CmdArgs[1]
	)

	mq, _ := g.MQs.GetMQ(ctx, cmd)
	job, _ := mq.Get(ctx, id)

	resp := struct {
		Status JobStatus        `json:"status"`
		Error  string           `json:"error,omitempty"`
		Result *json.RawMessage `json:"result,omitempty"`
	}{
		Status: job.Status,
		Error:  job.Error,
	}
	if job.Result != nil {
		msg := json.RawMessage(job.Result)
		resp.Result = &msg
	}

	return resp, nil
}

// POST /submit/<cmd>
func (g Gate) PostSubmit_(
	ctx context.Context,
	req *struct {
		CmdArgs []string
		ReqBody struct {
			ReqID   string      `json:"reqID"`
			HookURL string      `json:"hookURL"`
			Request interface{} `json:"request"`
		}
	},
	env *authstub.Env,
) (struct {
	JobID string `json:"job_id"`
}, error) {

	xl, ok := xlog.FromContext(ctx)
	if !ok {
		xl = xlog.NewWithReq(env.Req)
		ctx = xlog.NewContext(ctx, xl)
	}

	var (
		cmd = req.CmdArgs[0]
	)

	xl.Infof("submit request: %s", cmd)

	mq, _ := g.MQs.GetMQ(ctx, cmd)
	// TODO: use req.ReqBody instead?
	bs, _ := json.Marshal(req.ReqBody.Request)
	id, _ := mq.Submit(ctx, req.ReqBody.HookURL, bs, Env{UID: env.Uid, Utype: env.Utype})

	xl.Infof("new job: %s %s", cmd, id)

	return struct {
		JobID string `json:"job_id"`
	}{
		JobID: id,
	}, nil
}
