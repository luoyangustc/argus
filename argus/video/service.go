package video

import (
	"context"
	"net/http"
	"net/url"
	"strconv"
	"sync"
	"time"

	"github.com/qiniu/http/httputil.v1"
	xlog "github.com/qiniu/xlog.v1"
	authstub "qiniu.com/auth/authstub.v1"
)

type Service interface {
	PostVideo()
}

type service struct {
	Video
	OPs
	Jobs
	SaverHook
}

func NewService(
	ctx context.Context,
	video Video,
	ops OPs,
	jobs Jobs,
	saverHook SaverHook,
) service {
	// var evals = evals{ops: make(map[string]Eval)}
	// evals.ops[eFoo{}.OP()] = eFoo{}

	return service{
		Video:     video,
		OPs:       ops,
		Jobs:      jobs,
		SaverHook: saverHook,
	}
}

func (s service) PostVideo_(
	ctx context.Context,
	req *VideoRequest,
	env *authstub.Env,
) {

	var (
		id   = req.CmdArgs[0] // ID
		xl   = xlog.FromContextSafe(ctx)
		ends = struct {
			items map[string]struct {
				Labels   []ResultLabel   `json:"labels"`
				Segments []SegmentResult `json:"segments"`
			}
			sync.Mutex
		}{
			items: make(map[string]struct {
				Labels   []ResultLabel   `json:"labels"`
				Segments []SegmentResult `json:"segments"`
			}),
		}
		hooks = func(op string) EndHook {
			return EndHookFunc(
				func(ctx context.Context, rest EndResult) error {
					ends.Lock()
					ends.items[op] = rest.Result
					ends.Unlock()
					return nil
				},
			)
		}
		opParams    = make(map[string]OPParams)
		saverOPHook SaverOPHook
		err         error
	)

	if req.Ops == nil || len(req.Ops) == 0 {
		xl.Warnf("Empty OP Request.")
		httputil.ReplyErr(env.W, http.StatusBadRequest, "empty op")
		return
	}

	for _, op := range req.Ops {
		opParams[op.OP] = op.Params
	}
	if req.Params.Async {
		_, ok := s.OPs.Create(ctx, opParams, OPEnv{Uid: env.Uid, Utype: env.Utype})
		if !ok {
			xl.Warnf("Bad OP Request: %#v %#v", s.OPs, opParams)
			httputil.ReplyErr(env.W, http.StatusBadRequest, "bad op")
			return
		}
		// xl.Infof("OPS: %#v %#v", s.OPs, ops)
		jobID, _ := s.Jobs.Submit(ctx, env.Uid, env.Utype, id, *req)

		httputil.Reply(env.W, 200,
			struct {
				Job string `json:"job"`
			}{Job: jobID},
		)
		return
	}
	ops, ok := s.OPs.Create(ctx, opParams, OPEnv{Uid: env.Uid, Utype: env.Utype})
	if !ok {
		xl.Warnf("Bad OP Request: %#v %#v", s.OPs, opParams)
		httputil.ReplyErr(env.W, http.StatusBadRequest, "bad op")
		return
	}
	xl.Infof("OPS: %#v %#v", s.OPs, ops)

	requestsCounter("video_sync", "SB", "", "").Inc()
	requestsParallel("video_sync", "S").Inc()
	defer func(begin time.Time) {
		requestsParallel("video_sync", "S").Dec()
		responseTimeLong("video_sync", "S", "", "").
			Observe(durationAsFloat64(time.Since(begin)))
	}(time.Now())

	if s.SaverHook != nil && req.Params.Save != nil {
		saverOPHook, err = s.SaverHook.Get(ctx, env.Uid, id, *req.Params.Save)
		if err != nil {
			xl.Warnf("Saver %v", err)
		}
	}
	req.Data.URI, err = func(uri string, uid uint32) (string, error) {
		_url, err := url.Parse(uri)
		if err != nil {
			return uri, err
		}
		if _url.Scheme != "qiniu" {
			return uri, nil
		}
		_url.User = url.User(strconv.Itoa(int(uid)))
		return _url.String(), nil
	}(req.Data.URI, env.Uid)
	if err != nil {
		httputil.Error(env.W, err)
		return
	}
	err = s.Run(ctx, *req, ops, saverOPHook, hooks, nil, nil) // TODO
	if err != nil {
		httputil.Error(env.W, err)
		return
	}
	httputil.Reply(env.W, 200, ends.items)
	return
}

//----------------------------------------------------------------------------//

func (s service) GetJobsVideo_(
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

func (s service) GetJobsVideo(
	ctx context.Context,
	req *struct {
		Status string `json:"status"`
	},
	env *authstub.Env,
) ([]interface{}, error) {

	var (
		status = JobStatus(req.Status)
		// xl  = xlog.FromContextSafe(ctx)
	)

	jobs, _, err := s.Jobs.List(ctx, &env.Uid, &status, nil, nil, nil, nil) // TODO
	if err != nil {
		return nil, err
	}
	result := make([]interface{}, 0, len(jobs))
	for _, job := range jobs {
		result = append(result,
			struct {
				ID        string    `json:"id"`
				Status    JobStatus `json:"status"`
				CreatedAt time.Time `json:"created_at"`
				UpdatedAt time.Time `json:"updated_at"`
			}{
				ID:        job.ID,
				Status:    job.Status,
				CreatedAt: job.CreatedAt,
				UpdatedAt: job.UpdatedAt,
			})
	}
	return result, nil
}
