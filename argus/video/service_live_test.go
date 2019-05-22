package video

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	authstub "qiniu.com/auth/authstub.v1"
)

type mockVideo struct {
	CutResult
	SegmentResult
	EndResult
	t   *testing.T
	op  string
	err error
}

var _ Video = mockVideo{}

func (m mockVideo) Run(
	ctx context.Context,
	req VideoRequest,
	ops map[string]OP,
	saverOPHook SaverOPHook,
	endHook func(string) EndHook,
	cutHook func(string) CutHook,
	segmentHook func(string) SegmentHook,
) error {
	if m.op != "" {
		_, ok := ops[m.op]
		assert.True(m.t, ok)

		if end := endHook(m.op); end != nil {
			assert.Nil(m.t, end.End(ctx, m.EndResult))
		}

		if cut := cutHook(m.op); cut != nil {
			assert.Nil(m.t, cut.Cut(ctx, m.CutResult))
		}

		if segment := segmentHook(m.op); segment != nil {
			assert.Nil(m.t, segment.Segment(ctx, m.SegmentResult))
		}

	}
	return m.err
}

type mockJobs struct {
	t                      *testing.T
	Uid                    *uint32
	Status                 *JobStatus
	Marker                 *string
	Limit                  *int
	CreatedFrom, CreatedTo *time.Time
	JID                    string
}

var _ Jobs = mockJobs{}

func (m mockJobs) Submit(context.Context, uint32, uint32, string, VideoRequest) (string, error) {
	return "", nil
}

func (m mockJobs) Execute(context.Context, int, map[string]int) ([]Job, error) {
	return nil, nil
}

func (m mockJobs) Touch(context.Context, *Job) error {
	return nil
}

func (m mockJobs) Finish(context.Context, Job) error {
	return nil
}

func (m mockJobs) Cancel(context.Context, uint32, string) error {
	return nil
}

func (m mockJobs) Cancelled(context.Context, Job) error {
	return nil
}

func (m mockJobs) Get(ctx context.Context, uid uint32, jid string) (Job, error) {
	if m.Uid != nil {
		assert.Equal(m.t, *m.Uid, uid)
	}
	fmt.Println(m.JID, jid)
	assert.Equal(m.t, m.JID, jid)
	return Job{}, nil
}

func (m mockJobs) List(ctx context.Context, uid *uint32, status *JobStatus, created_from, created_to *time.Time, marker *string, limit *int) ([]Job, string, error) {
	if m.Uid != nil {
		assert.Equal(m.t, *m.Uid, *uid)
	} else {
		assert.Nil(m.t, uid)
	}
	if m.Status != nil {
		assert.Equal(m.t, *m.Status, *status)
	} else {
		assert.Nil(m.t, status)
	}
	if m.CreatedFrom != nil {
		// check by second
		assert.Equal(m.t, m.CreatedFrom.Unix(), created_from.Unix())
	} else {
		assert.Nil(m.t, created_from)
	}
	if m.CreatedTo != nil {
		assert.Equal(m.t, m.CreatedTo.Unix(), created_to.Unix())
	} else {
		assert.Nil(m.t, created_to)
	}
	if m.Marker != nil {
		assert.Equal(m.t, *m.Marker, *marker)
	} else {
		assert.Nil(m.t, marker)
	}
	if m.Limit != nil {
		assert.Equal(m.t, *m.Limit, *limit)
	} else {
		assert.Nil(m.t, limit)
	}
	return nil, "", nil
}

type mockOPs struct{}

var _ OPs = mockOPs{}

func (m mockOPs) Load() map[string]OP       { return map[string]OP{"foo": eFoo{}} }
func (m mockOPs) ResetOP(string, *OPConfig) {}
func (m mockOPs) Create(ctx context.Context, ops map[string]OPParams, env OPEnv) (map[string]OP, bool) {
	ret := make(map[string]OP, 0)
	for op, _ := range ops {
		if op == "foo" {
			ret["foo"] = eFoo{}
		}
	}
	return ret, true
}

type mockSaveHook struct{}

var _ SaverHook = mockSaveHook{}

func (m mockSaveHook) Get(ctx context.Context, uid uint32, vid string, params json.RawMessage) (SaverOPHook, error) {
	return nil, nil
}

func mockServer() *httptest.Server {
	f := func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(200)
	}
	return httptest.NewServer(http.HandlerFunc(f))
}

///////////////////////////////////////////////////////////////////////////////////

func TestGetJobsLive(t *testing.T) {
	var (
		ctx   = context.Background()
		video mockVideo
		ops   mockOPs
		jobs                = mockJobs{t: t}
		env   *authstub.Env = &authstub.Env{}
		err   error
		req   = &struct {
			CmdArgs     []string
			Status      string `json:"status,omitempty"`
			CreatedFrom int64  `json:"created_from,omitempty"`
			CreatedTo   int64  `json:"created_to,omitempty"`
			Marker      string `json:"marker,omitempty"`
			Limit       int    `json:"limit,omitempty"`
		}{}
		uid uint32 = 1234
	)
	env.Uid = uid
	svr := NewServiceLive(ctx, video, ops, jobs, nil, 5.0)

	t.Run("默认查询", func(t *testing.T) {
		{
			status := JobStatus("")
			jobs.Uid = &uid
			jobs.Status = &status
			jobs.CreatedFrom, jobs.CreatedTo = nil, nil
			marker := ""
			jobs.Marker = &marker
			limit := 1000
			jobs.Limit = &limit
			svr.Jobs = jobs
		}

		_, err = svr.GetJobsLive(ctx, req, env)
		assert.Nil(t, err)
	})

	t.Run("按marker和limit", func(t *testing.T) {
		{
			status := JobStatus("")
			jobs.Uid = &uid
			jobs.Status = &status
			jobs.CreatedFrom, jobs.CreatedTo = nil, nil
			marker := "test_mark"
			jobs.Marker = &marker
			limit := 1000
			jobs.Limit = &limit
			svr.Jobs = jobs
			req.Marker = marker
		}

		_, err = svr.GetJobsLive(ctx, req, env)
		assert.Nil(t, err)

		{
			limit := 500
			jobs.Limit = &limit
			svr.Jobs = jobs
			req.Limit = 500
		}
		_, err = svr.GetJobsLive(ctx, req, env)
		assert.Nil(t, err)

		{
			limit := 1000
			jobs.Limit = &limit
			svr.Jobs = jobs
			req.Limit = -1
		}
		_, err = svr.GetJobsLive(ctx, req, env)
		assert.Nil(t, err)

		{
			limit := 1000
			jobs.Limit = &limit
			svr.Jobs = jobs
			req.Limit = 1001
		}
		_, err = svr.GetJobsLive(ctx, req, env)
		assert.Nil(t, err)
	})

	t.Run("按marker和limit", func(t *testing.T) {
		{
			status := JobStatus("DOING")
			jobs.Uid = &uid
			jobs.Status = &status
			jobs.CreatedFrom, jobs.CreatedTo = nil, nil
			marker := ""
			jobs.Marker = &marker
			limit := 1000
			jobs.Limit = &limit
			svr.Jobs = jobs
			req.Limit = limit
			req.Marker = marker
			req.Status = string(status)
		}
		_, err = svr.GetJobsLive(ctx, req, env)
		assert.Nil(t, err)
	})

	t.Run("按marker和limit", func(t *testing.T) {
		{
			status := JobStatus("")
			tt := time.Now()
			jobs.Uid = &uid
			jobs.Status = &status
			jobs.CreatedFrom, jobs.CreatedTo = &tt, &tt
			marker := ""
			jobs.Marker = &marker
			limit := 1000
			jobs.Limit = &limit
			svr.Jobs = jobs
			req.Status = string("")
			req.CreatedFrom, req.CreatedTo = tt.Unix(), tt.Unix()
		}
		_, err = svr.GetJobsLive(ctx, req, env)
		assert.Nil(t, err)

		{
			jobs.CreatedFrom, jobs.CreatedTo = nil, nil
			svr.Jobs = jobs
			req.CreatedFrom, req.CreatedTo = -1, -1
		}
		_, err = svr.GetJobsLive(ctx, req, env)
		assert.Nil(t, err)
	})
}

func TestGetJobsLive_(t *testing.T) {
	var (
		ctx   = context.Background()
		video mockVideo
		ops   mockOPs
		saver mockSaveHook
		jobs                = mockJobs{t: t}
		env   *authstub.Env = &authstub.Env{}
		err   error
		req   = &struct {
			CmdArgs []string
		}{}
		uid    uint32 = 1234
		job_id        = "job001"
	)
	env.Uid = uid
	svr := NewServiceLive(ctx, video, ops, jobs, saver, 5.0)
	req.CmdArgs = append(req.CmdArgs, job_id)
	{
		jobs.Uid = &uid
		jobs.JID = job_id
		svr.Jobs = jobs
	}
	_, err = svr.GetJobsLive_(ctx, req, env)
	assert.Nil(t, err)
}

func TestRunLive(t *testing.T) {
	server := mockServer()
	defer server.Close()
	var (
		ctx           = context.Background()
		url    string = server.URL
		liveID string = "live_id_001"
		jobID  string = "job_id_001"
		opName string = "foo"
		uid    uint32 = 1234
		env    OPEnv  = OPEnv{Uid: uid}
		req    VideoRequest
		video  mockVideo = mockVideo{t: t}
		ops    mockOPs
		jobs                 = mockJobs{t: t}
		auth   *authstub.Env = &authstub.Env{}
	)
	auth.Uid = uid

	req.CmdArgs = []string{liveID}
	op := struct {
		OP             string   `json:"op"`
		CutHookURL     string   `json:"cut_hook_url"`
		SegmentHookURL string   `json:"segment_hook_url"`
		HookURL        string   `json:"hookURL"`
		Params         OPParams `json:"params"`
	}{
		OP:             opName,
		CutHookURL:     url,
		SegmentHookURL: url,
		HookURL:        url,
	}
	req.Ops = append(req.Ops, op)

	svr := NewServiceLive(ctx, video, ops, jobs, nil, 5.0)
	assert.Nil(t, svr.RunLive(ctx, req, jobID, env))

	video.op = opName
	svr.Video = video
	assert.Nil(t, svr.RunLive(ctx, req, jobID, env))

	e1 := errors.New("test_error")
	video.err = e1
	svr.Video = video
	req.Params.HookURL = url
	assert.Equal(t, e1, svr.RunLive(ctx, req, jobID, env))
}
