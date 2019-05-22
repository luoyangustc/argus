package video

import (
	"context"
	"net/http"
	"time"

	httputil "github.com/qiniu/http/httputil.v1"
	rpc "github.com/qiniu/rpc.v3"
	xlog "github.com/qiniu/xlog.v1"
	authstub "qiniu.com/auth/authstub.v1"
)

const (
	MAX_GET_JOBS_LIVE_LIMIT int = 1000
)

type LiveService interface {
	RunLive(context.Context, VideoRequest, string, OPEnv) error
}

type serviceLive struct {
	Video
	OPs
	Jobs
	SaverHook
	DefaultLiveTimeout float64
}

func NewServiceLive(
	ctx context.Context,
	video Video,
	ops OPs,
	jobs Jobs,
	saverHook SaverHook,
	defaultLiveTimeout float64,
) serviceLive {
	// var evals = evals{ops: make(map[string]Eval)}
	// evals.ops[eFoo{}.OP()] = eFoo{}

	return serviceLive{
		Video:              video,
		OPs:                ops,
		Jobs:               jobs,
		SaverHook:          saverHook,
		DefaultLiveTimeout: defaultLiveTimeout,
	}
}

func (s serviceLive) PostLive_(
	ctx context.Context,
	req *VideoRequest,
	env *authstub.Env,
) {

	var (
		id = req.CmdArgs[0] // ID
		xl = xlog.FromContextSafe(ctx)

		opParams = make(map[string]OPParams)
	)

	if req.Ops == nil || len(req.Ops) == 0 {
		xl.Warnf("Empty OP Request.")
		httputil.ReplyErr(env.W, http.StatusBadRequest, "empty op")
		return
	}

	if req.Params.Live == nil {
		req.Params.Live = &struct {
			Timeout    float64 `json:"timeout"`
			Downstream string  `json:"downstream"`
		}{
			Timeout: s.DefaultLiveTimeout,
		}
	} else if req.Params.Live.Timeout <= 0 {
		req.Params.Live.Timeout = s.DefaultLiveTimeout
	}

	for _, op := range req.Ops {
		opParams[op.OP] = op.Params
	}
	ops, ok := s.OPs.Create(ctx, opParams, OPEnv{Uid: env.Uid, Utype: env.Utype})
	if !ok {
		xl.Warnf("Bad OP Request: %#v %#v", s.OPs, opParams)
		httputil.ReplyErr(env.W, http.StatusBadRequest, "bad op")
		return
	}
	defer func() {
		for _, op := range ops {
			_ = op.Reset(ctx)
		}
	}()
	// xl.Infof("OPS: %#v %#v", s.OPs, ops)
	jobID, _ := s.Jobs.Submit(ctx, env.Uid, env.Utype, id, *req)

	httputil.Reply(env.W, 200,
		struct {
			Job string `json:"job"`
		}{
			Job: jobID,
		},
	)
	return
}

func (s serviceLive) PostJobs_Kill(
	ctx context.Context,
	req *struct {
		CmdArgs []string
	},
	env *authstub.Env,
) {
	var (
		id  = req.CmdArgs[0] // ID
		xl  = xlog.FromContextSafe(ctx)
		err error
	)

	err = s.Jobs.Cancel(ctx, env.Uid, id)
	if err != nil {
		xl.Errorf("fail to cancel job %s, err: %s", id, err)
		httputil.ReplyErr(env.W, http.StatusBadRequest, "fail to cancel job")
	} else {
		httputil.Reply(env.W, 200, struct{}{})
	}
	return
}

func (s serviceLive) GetJobsLive(
	ctx context.Context,
	req *struct {
		CmdArgs     []string
		Status      string `json:"status,omitempty"`
		CreatedFrom int64  `json:"created_from,omitempty"`
		CreatedTo   int64  `json:"created_to,omitempty"`
		Marker      string `json:"marker,omitempty"`
		Limit       int    `json:"limit,omitempty"`
	},
	env *authstub.Env,
) (interface{}, error) {

	var (
		status                 = JobStatus(req.Status)
		err                    error
		jobs                   []Job
		createdFrom, createdTo *time.Time
	)

	type _result struct {
		ID        string    `json:"id"`
		Live      string    `json:"live"`
		Status    JobStatus `json:"status"`
		CreatedAt time.Time `json:"created_at"`
		UpdatedAt time.Time `json:"updated_at"`
	}
	var result struct {
		Jobs   []_result `json:"jobs"`
		Marker string    `json:"marker,omitempty"`
	}

	// created_from, created_to: unix timestamp, by second
	if req.CreatedFrom > 0 {
		from := time.Unix(req.CreatedFrom, 0)
		createdFrom = &from
	}
	if req.CreatedTo > 0 {
		from := time.Unix(req.CreatedTo, 0)
		createdTo = &from
	}

	if req.Limit <= 0 || req.Limit > MAX_GET_JOBS_LIVE_LIMIT {
		req.Limit = MAX_GET_JOBS_LIVE_LIMIT
	}

	jobs, result.Marker, err = s.Jobs.List(ctx, &env.Uid, &status, createdFrom, createdTo, &req.Marker, &req.Limit)
	if err != nil {
		return nil, err
	}
	result.Jobs = make([]_result, 0, len(jobs))
	for _, job := range jobs {
		result.Jobs = append(result.Jobs,
			_result{
				ID:        job.ID,
				Live:      job.VID,
				Status:    job.Status,
				CreatedAt: job.CreatedAt,
				UpdatedAt: job.UpdatedAt,
			})
	}

	return result, nil
}

func (s serviceLive) GetJobsLive_(
	ctx context.Context,
	req *struct {
		CmdArgs []string
	},
	env *authstub.Env,
) (Job, error) {

	var (
		jid = req.CmdArgs[0]
		xl  = xlog.FromContextSafe(ctx)
	)

	xl.Infof("query job: %s", jid)
	job, err := s.Jobs.Get(ctx, env.Uid, jid)
	return job, err
}

func (s serviceLive) notify(ctx context.Context, url string, body interface{}) (err error) {
	var xl = xlog.FromContextSafe(ctx)
	defer func() {
		xl.Debugf("notify done. %s %v", url, err)
	}()
	xl.Debugf("try notify: %s", url)
	err = rpc.DefaultClient.CallWithJson(ctx, nil, "POST", url, body)
	return
}

func (s serviceLive) RunLive(ctx context.Context, req VideoRequest, jobId string, env OPEnv) error {
	xlog.FromContextSafe(ctx).Info("Run Live....")
	type (
		LiveEndResult struct {
			LiveID  string               `json:"live_id,omitempty"`
			JobID   string               `json:"job_id,omitempty"`
			Code    int                  `json:"code,omitempty"`
			Message string               `json:"message,omitempty"`
			Result  map[string]EndResult `json:"result,omitempty"`
		}

		endResult struct {
			LiveID string `json:"live_id,omitempty"`

			Code        int    `json:"code"`
			Message     string `json:"message"`
			OP          string `json:"op,omitempty"`
			OffsetBegin int64  `json:"offset_begin,omitempty"`
			OffsetEnd   int64  `json:"offset_end,omitempty"`
			Result      struct {
				Labels   []ResultLabel   `json:"labels,omitempty"`
				Segments []SegmentResult `json:"segments"`
			} `json:"result"`
		}

		segmentResult struct {
			LiveID string `json:"live_id,omitempty"`

			OP          string        `json:"op,omitempty"`
			OffsetBegin int64         `json:"offset_begin"`
			OffsetEnd   int64         `json:"offset_end"`
			Labels      []ResultLabel `json:"labels,omitempty"`
			Cuts        []CutResult   `json:"cuts,omitempty"`
			Clips       []ClipResult  `json:"clips,omitempty"`
		}

		cutResult struct {
			JobID  string `json:"job_id"`
			LiveID string `json:"live_id,omitempty"`
			OP     string `json:"op,omitempty"`

			Offset int64       `json:"offset"`
			URI    string      `json:"uri,omitempty"`
			Result interface{} `json:"result"`
		}
	)

	var (
		id = req.CmdArgs[0] // ID
		xl = xlog.FromContextSafe(ctx)

		// liveEnd = LiveEndResult{} // TODO

		endHooks = func(op string) EndHook {
			return EndHookFunc(func(ctx context.Context, ret EndResult) error {
				return nil
			})
		}
		segmentHooks = func(op string) SegmentHook {
			var url string
			for _, params := range req.Ops {
				if params.OP == op {
					url = params.SegmentHookURL
				}
			}
			return segmentHookFunc(func(ctx context.Context, ret SegmentResult) error {
				if url == "" {
					return nil
				}
				ret0 := segmentResult{
					LiveID: id, OP: op,
					OffsetBegin: ret.OffsetBegin, OffsetEnd: ret.OffsetEnd,
					Labels: ret.Labels, Cuts: ret.Cuts, Clips: ret.Clips,
				}
				return s.notify(ctx, url, ret0)
			})
		}
		cutHooks = func(op string) CutHook {
			var url string
			for _, params := range req.Ops {
				if params.OP == op {
					url = params.CutHookURL
				}
			}
			return CutHookFunc(func(ctx context.Context, ret CutResult) error {
				if url == "" {
					return nil
				}
				ret0 := cutResult{
					LiveID: id, JobID: jobId, OP: op,
					Offset: ret.Offset, URI: ret.URI, Result: ret.Result,
				}
				return s.notify(ctx, url, ret0)
			})
		}

		opParams    = make(map[string]OPParams)
		saverOPHook SaverOPHook
		err         error
	)

	if req.Ops == nil || len(req.Ops) == 0 {
		xl.Warnf("Empty OP Request.")
		return httputil.NewError(http.StatusBadRequest, "empty op")
	}

	for _, op := range req.Ops {
		opParams[op.OP] = op.Params
	}
	ops, ok := s.OPs.Create(ctx, opParams, OPEnv{Uid: env.Uid, Utype: env.Utype})
	if !ok {
		xl.Warnf("Bad OP Request: %#v %#v", s.OPs, opParams)
		return httputil.NewError(http.StatusBadRequest, "bad op")
	}
	xl.Infof("OPS: %#v %#v", s.OPs, ops)

	defer func() {
		for name, op := range ops {
			if e := op.Reset(ctx); e != nil {
				xl.Warnf("reset op (%s) failed, error: %s", name, e.Error())
			}
		}
	}()

	if s.SaverHook != nil && req.Params.Save != nil {
		saverOPHook, err = s.SaverHook.Get(ctx, env.Uid, id, *req.Params.Save)
		if err != nil {
			xl.Warnf("Saver %v", err)
		}
	}
	err = s.Run(ctx, req, ops, saverOPHook, endHooks, cutHooks, segmentHooks)

	if err != nil && req.Params.HookURL != "" {
		r := LiveEndResult{
			LiveID:  id,
			JobID:   jobId,
			Code:    http.StatusInternalServerError,
			Message: err.Error(),
		}
		err := rpc.DefaultClient.CallWithJson(ctx, nil, "POST", req.Params.HookURL, r)
		xl.Infof("callback done. %v %s", err, req.Params.HookURL)
	}

	return err

}
