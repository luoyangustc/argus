package live

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"net/http"
	"strings"
	"time"

	httputil "github.com/qiniu/http/httputil.v1"
	rpc "github.com/qiniu/rpc.v3"
	xlog "github.com/qiniu/xlog.v1"

	"qiniu.com/argus/sdk/video"
	. "qiniu.com/argus/service/service"
	svideo "qiniu.com/argus/service/service/video"
	video0 "qiniu.com/argus/video"
	"qiniu.com/argus/video/vframe"
)

var _ Service = service{}

type Service interface {
	Live(context.Context, video0.VideoRequest, string) error
	Async(context.Context, video0.VideoRequest) (interface{}, error)
	Kill(context.Context, string) (interface{}, error)
	GetJobs(context.Context, GetJobsRequest) (interface{}, error)
	GetJobByID(context.Context, string) (video0.Job, error)
}

type Config struct {
	DefaultVframeParams vframe.VframeParams `json:"default_vframe"`
	Workerspace         string              `json:"-"`
	SaverHook           svideo.SaverHook    `json:"-"`
}

type service struct {
	Config
	OPs
	video0.Jobs
}

func NewService(
	ctx context.Context,
	config Config,
	ops OPs,
	jobs video0.Jobs,
) service {
	return service{
		Config: config,
		OPs:    ops,
		Jobs:   jobs,
	}
}

func (s service) live(ctx context.Context, req video0.VideoRequest, jobID string) error {
	var (
		vid       = req.CmdArgs[0]
		vframeReq vframe.VframeRequest
		saver     svideo.Saver
	)

	{
		if req.Params.Vframe == nil || req.Params.Vframe.Mode == nil {
			req.Params.Vframe = &s.DefaultVframeParams
		}
		if req.Params.Vframe.GetMode() == 2 &&
			req.Params.Vframe.Interval == 0 {
			req.Params.Vframe.Interval = 25
		}
		vframeReq.Data.URI = req.Data.URI
		vframeReq.Params = *req.Params.Vframe
		if req.Params.Live != nil {
			vframeReq.Live = &vframe.LiveParams{
				Timeout:   req.Params.Live.Timeout,
				Downsteam: req.Params.Live.Downstream,
			}
		} else {
			vframeReq.Live = &vframe.LiveParams{}
		}
	}
	{
		if req.Params.Vframe.GetMode() != 2 {
			return httputil.NewError(
				http.StatusBadRequest,
				"invalid mode, allow mode is [2]")
		}
		if req.Params.Vframe.GetMode() == 2 {
			if req.Params.Vframe.Interval <= 0 {
				return httputil.NewError(
					http.StatusBadRequest,
					"invalid interval, allow interval is [1, ~]")
			}
		}
	}
	if s.SaverHook != nil && req.Params.Save != nil {
		saver, _ = s.SaverHook.Get(ctx, vid, *req.Params.Save)
	}

	type _OP struct {
		CutHook  video0.CutHook
		OPParams video0.OPParams
	}

	var (
		ops   = make(map[string]_OP)
		pipes = make(map[string]video.CutsPipe)
		call  = func(ctx context.Context, op, url string, body interface{}) error {
			var (
				xl = xlog.FromContextSafe(ctx)
			)
			xl.Debugf("try to call: %s %s", op, url)
			err := rpc.DefaultClient.CallWithJson(ctx, nil, "POST", url, body)
			xl.Debugf("callback done. %s %v %s", op, err, url)
			return err
		}
	)
	for _, op := range req.Ops {
		of, ok := s.OPs[op.OP]
		if !ok {
			return formatError(ErrArgs("bad op"))
		}
		_op, err := of.(svideo.OPFactory).Create(ctx, op.Params)
		if err != nil {
			return formatError(err)
		}
		cutOP, ok1 := _op.(CutOP)
		if !ok1 {
			continue
		}
		pipes[op.OP], err = cutOP.NewCuts(ctx)
		if err != nil {
			return formatError(err)
		}
		ops[op.OP] = _OP{}
		if len(op.CutHookURL) > 0 {
			var (
				name = op.OP
				url  = op.CutHookURL
			)
			cutHook := video0.CutHookFunc(func(ctx context.Context, ret video0.CutResult) error {
				ret0 := struct {
					JobID  string `json:"job_id"`
					LiveID string `json:"live_id,omitempty"`
					OP     string `json:"op,omitempty"`

					Offset int64       `json:"offset"`
					URI    string      `json:"uri,omitempty"`
					Result interface{} `json:"result"`
				}{
					LiveID: vid,
					JobID:  jobID,
					OP:     name,
					Offset: ret.Offset,
					URI:    ret.URI,
					Result: ret.Result,
				}
				return call(ctx, name, url, ret0)
			})
			_op := ops[op.OP]
			_op.CutHook = cutHook
			_op.OPParams = op.Params
			ops[op.OP] = _op
		}
	}

	xlog.FromContextSafe(ctx).Debugf("ops: %v | %v | %v", req.Ops, ops, s.OPs)

	ctx, cancel := context.WithCancel(ctx)
	defer cancel()
	vf := video.NewFfmpegai(ctx,
		vframeReq.Data.URI,
		int64(vframeReq.Params.Interval),
		int64(vframeReq.Live.Timeout*1000),
		vframeReq.Live.Downsteam,
	)
	err := video.Video(ctx, vf, video.CreateMultiCutsPipe(pipes),
		func(ctx context.Context, resp video.CutResponse) {
			xl := xlog.FromContextSafe(ctx)

			xl.Debugf("begin cut. %v", resp.OffsetMS)

			rets := resp.Result.(map[string]video.CutResponse)
			rets1 := make(map[string]video0.CutResultWithLabels)
			var uri string
			for name, ret := range rets {
				if ret.Error != nil {
					xl.Warnf("ret err: %v %s %v", ret.OffsetMS, name, ret.Error)
					// TODO
					continue
				}
				op := ops[name]
				ret1 := ret.Result.(video0.CutResultWithLabels)
				var ok bool
				ok, ret1 = video0.SelectCut(
					op.OPParams.Labels,
					op.OPParams.IgnoreEmptyLabels,
					ret1)
				if !ok {
					continue
				}
				if saver != nil && uri == "" {
					req := vf.Get(ctx, ret.OffsetMS)
					if req.Body != nil {
						uri, _ = saver.Save(ctx, ret.OffsetMS,
							"data:application/octet-stream;base64,"+
								base64.StdEncoding.EncodeToString(req.Body),
						)
						xl.Debugf("save: %v %s", ret.OffsetMS, uri)
					}
				}
				if op.CutHook != nil {
					_ = op.CutHook.Cut(ctx, video0.CutResult{
						Offset: ret.OffsetMS,
						URI:    uri,
						Result: ret1,
					})
				}
				rets1[name] = ret1
			}
			_ = vf.Set(ctx, video.CutResponse{OffsetMS: resp.OffsetMS, Result: rets1})
		})
	if err != nil {
		return formatError(err)
	}
	if req.Params.HookURL != "" {
		var (
			message string
		)
		if err != nil {
			message = err.Error()
		}
		err = call(ctx, "", req.Params.HookURL, struct {
			ID     string                      `json:"id"`
			JobID  string                      `json:"job_id,omitempty"`
			Error  string                      `json:"error,omitempty"`
			Meta   json.RawMessage             `json:"meta,omitempty"`
			Result map[string]video0.EndResult `json:"result,omitempty"`
		}{
			ID:    vid,
			JobID: jobID,
			Error: message,
			Meta:  req.Data.Attribute.Meta,
		})
	}
	return formatError(err)
}

func (s service) Live(ctx context.Context, req video0.VideoRequest, jobID string) error {

	var (
		id = req.CmdArgs[0] // ID
		xl = xlog.FromContextSafe(ctx)
	)

	if req.Ops == nil || len(req.Ops) == 0 {
		xl.Warnf("Empty OP Request. %s", id)
		return ErrArgs("empty op")
	}

	for _, op := range req.Ops {
		if _, ok := s.OPs[op.OP]; !ok {
			xl.Warnf("Bad OP Request: %#v %#v", s.OPs, op)
			return ErrArgs("bad op")
		}
	}

	return s.live(ctx, req, jobID)
}

const (
	DefaultUserID   = 1
	DefaultUserType = 0
)

func (s service) Async(ctx context.Context, req video0.VideoRequest) (interface{}, error) {
	var (
		id  = req.CmdArgs[0] // live_id
		xl  = xlog.FromContextSafe(ctx)
		end struct {
			Job string `json:"job"`
		}
	)

	if req.Params.Live != nil {
		if req.Params.Live.Timeout < 0 {
			return nil, ErrArgs("bad live timeout")
		}
		if len(req.Params.Live.Downstream) > 0 {
			if !strings.HasPrefix(req.Params.Live.Downstream, "rtsp://") &&
				!strings.HasPrefix(req.Params.Live.Downstream, "rtmp://") {
				return nil, ErrArgs("bad downstream, onle support rtsp/rtmp protocol")
			}
		}
	}

	if req.Ops == nil || len(req.Ops) == 0 {
		xl.Warnf("Empty OP Request. %s", id)
		return nil, ErrArgs("empty op")
	}

	for _, op := range req.Ops {
		opf, ok := s.OPs[op.OP]
		if !ok {
			xl.Warnf("Bad OP Request: %#v %#v", s.OPs, op)
			return nil, ErrArgs("bad op")
		}
		_op, err := opf.Create(ctx, op.Params)
		if err != nil {
			xl.Warnf("create op failed: %#v, error: %s", op.Params, err.Error())
			return nil, err
		}
		opf.Release(ctx, _op)
	}

	end.Job, _ = s.Jobs.Submit(ctx, DefaultUserID, DefaultUserType, id, req)

	return end, nil
}

func (s service) Kill(ctx context.Context, id string) (interface{}, error) {
	var (
		xl = xlog.FromContextSafe(ctx)
	)
	if err := s.Jobs.Cancel(ctx, DefaultUserID, id); err != nil {
		xl.Errorf("fail to cancel job %s, err: %s", id, err)
		return nil, ErrAsyncJob(err.Error())
	} else {
		xl.Infof("Cancelling job %s", id)
		return struct{}{}, nil
	}
}

type GetJobsRequest struct {
	Status      string `json:"status,omitempty"`
	CreatedFrom int64  `json:"created_from,omitempty"`
	CreatedTo   int64  `json:"created_to,omitempty"`
	Marker      string `json:"marker,omitempty"`
	Limit       int    `json:"limit,omitempty"`
}

type GetJobsResult struct {
	ID        string           `json:"id"`
	JobID     string           `json:"job_id"`
	Status    video0.JobStatus `json:"status"`
	CreatedAt time.Time        `json:"created_at"`
	UpdatedAt time.Time        `json:"updated_at"`
}

type GetJobsResponse struct {
	Jobs   []GetJobsResult `json:"jobs"`
	Marker string          `json:"marker,omitempty"`
}

func (s service) GetJobs(ctx context.Context, req GetJobsRequest) (interface{}, error) {

	var (
		result                 GetJobsResponse
		status                 = video0.JobStatus(req.Status)
		err                    error
		jobs                   []video0.Job
		createdFrom, createdTo *time.Time
		uid                    uint32 = uint32(DefaultUserID)
	)
	// created_from, created_to: unix timestamp, by second
	if req.CreatedFrom > 0 {
		from := time.Unix(req.CreatedFrom, 0)
		createdFrom = &from
	}
	if req.CreatedTo > 0 {
		from := time.Unix(req.CreatedTo, 0)
		createdTo = &from
	}

	if req.Limit <= 0 || req.Limit > video0.MAX_GET_JOBS_LIVE_LIMIT {
		req.Limit = video0.MAX_GET_JOBS_LIVE_LIMIT
	}
	jobs, result.Marker, err = s.Jobs.List(
		ctx, &uid, &status,
		createdFrom, createdTo,
		&req.Marker, &req.Limit)
	if err != nil {
		return nil, ErrAsyncJob(err.Error())
	}
	result.Jobs = make([]GetJobsResult, 0, len(jobs))
	for _, job := range jobs {
		result.Jobs = append(result.Jobs,
			GetJobsResult{
				ID:        job.VID,
				JobID:     job.ID,
				Status:    job.Status,
				CreatedAt: job.CreatedAt,
				UpdatedAt: job.UpdatedAt,
			})
	}
	return result, nil
}
func (s service) GetJobByID(ctx context.Context, jobID string) (video0.Job, error) {
	job, err := s.Jobs.Get(ctx, DefaultUserID, jobID)
	if err != nil {
		return job, ErrAsyncJob(err.Error())
	}
	return job, nil
}
