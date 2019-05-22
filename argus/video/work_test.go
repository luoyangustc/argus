package video

import (
	"context"
	"encoding/json"
	"sync"
	"testing"
	"time"

	mgoutil "github.com/qiniu/db/mgoutil.v3"
	"github.com/stretchr/testify/assert"
	mgo "gopkg.in/mgo.v2"
)

func TestJobResultInMgo(t *testing.T) {

	var (
		MGO_HOST = "mongodb://127.0.0.1"
		MGO_DB   = "argus_video_test"
		cfg      = JobsInMgoConfig{
			Mgo: mgoutil.Config{
				Host: "127.0.0.1:27017",
				DB:   MGO_DB,
			},
			MgoPoolLimit:   100,
			IdleJobTimeout: 100000000, // 100ms = 10^8 ns
		}
		ctx          = context.Background()
		uid   uint32 = 2002
		utype uint32 = 2
		jobs  _JobsInMgo
	)

	sess, err := mgo.Dial(MGO_HOST)
	assert.NoError(t, err)
	defer func() {
		assert.Nil(t, sess.DB(MGO_DB).DropDatabase())
	}()
	jobs, err = NewJobsInMgo(cfg)
	assert.NoError(t, err)
	assert.NotNil(t, jobs)

	type Msg struct {
		A int `json:"b"`
	}

	t.Run("json.RawMessage", func(t *testing.T) {
		id, err := jobs.Submit(ctx, uid, utype, "V001", VideoRequest{})
		assert.NoError(t, err)
		job, err := jobs.Get(ctx, uid, id)
		assert.NoError(t, err)
		job.Result, _ = json.Marshal(Msg{A: 10})
		err = jobs.Finish(ctx, job)
		assert.NoError(t, err)
		job, err = jobs.Get(ctx, uid, id)
		assert.NoError(t, err)

		bs, _ := json.Marshal(job)
		_ = json.Unmarshal(bs, &job)
		var msg Msg
		_ = json.Unmarshal(job.Result, &msg)
		assert.Equal(t, 10, msg.A)
	})
	t.Run("json.RawMessage Error", func(t *testing.T) {
		id, err := jobs.Submit(ctx, uid, utype, "V002", VideoRequest{})
		assert.NoError(t, err)
		job, err := jobs.Get(ctx, uid, id)
		assert.NoError(t, err)
		job.Result = json.RawMessage("Result")
		job.Error = "XX"
		err = jobs.Finish(ctx, job)
		assert.NoError(t, err)
		job, err = jobs.Get(ctx, uid, id)
		assert.NoError(t, err)

		bs, _ := json.Marshal(job)
		_ = json.Unmarshal(bs, &job)
		assert.Equal(t, "XX", job.Error)
		assert.Equal(t, json.RawMessage("Result"), job.Result)
	})
	t.Run("OLD interface{}", func(t *testing.T) {
		id, err := jobs.Submit(ctx, uid, utype, "V003", VideoRequest{})
		assert.NoError(t, err)
		job, err := jobs.Get(ctx, uid, id)
		assert.NoError(t, err)

		_job := job.Ref.(_JobInMgo)
		_job.Result = map[string]EndResult{"P1": EndResult{Result: struct {
			Labels   []ResultLabel   `json:"labels"`
			Segments []SegmentResult `json:"segments"`
		}{Labels: []ResultLabel{ResultLabel{Name: "YY"}}}}}
		func() {
			coll := jobs.coll.CopySession()
			defer coll.CloseSession()
			err = jobs.update(ctx, coll, &_job)
			assert.NoError(t, err)
		}()
		job, err = jobs.Get(ctx, uid, id)
		assert.NoError(t, err)

		bs, _ := json.Marshal(job)
		_ = json.Unmarshal(bs, &job)
		var m = map[string]EndResult{}
		_ = json.Unmarshal(job.Result, &m)
		assert.Equal(t, 1, len(m))
		assert.Equal(t, 1, len(m["P1"].Result.Labels))
		assert.Equal(t, "YY", m["P1"].Result.Labels[0].Name)
	})
	t.Run("OLD interface{} Error", func(t *testing.T) {
		id, err := jobs.Submit(ctx, uid, utype, "V002", VideoRequest{})
		assert.NoError(t, err)
		job, err := jobs.Get(ctx, uid, id)
		assert.NoError(t, err)
		_job := job.Ref.(_JobInMgo)
		_job.Error = "XX"
		func() {
			coll := jobs.coll.CopySession()
			defer coll.CloseSession()
			err = jobs.update(ctx, coll, &_job)
			assert.NoError(t, err)
		}()
		job, err = jobs.Get(ctx, uid, id)
		assert.NoError(t, err)

		bs, _ := json.Marshal(job)
		_ = json.Unmarshal(bs, &job)
		assert.Equal(t, "XX", job.Error)
		assert.Nil(t, job.Result)
	})

}

func TestJobsInMgo(t *testing.T) {
	var (
		MGO_HOST = "mongodb://127.0.0.1"
		MGO_DB   = "argus_video_test"
		cfg      = JobsInMgoConfig{
			Mgo: mgoutil.Config{
				Host: "127.0.0.1:27017",
				DB:   MGO_DB,
			},
			MgoPoolLimit:   100,
			IdleJobTimeout: 100000000, // 100ms = 10^8 ns
		}
		ctx          = context.Background()
		uid   uint32 = 2002
		utype uint32 = 2
		jobs  Jobs
		ops   = make([]struct {
			OP             string   `json:"op"`
			CutHookURL     string   `json:"cut_hook_url"`
			SegmentHookURL string   `json:"segment_hook_url"`
			HookURL        string   `json:"hookURL"`
			Params         OPParams `json:"params"`
		}, 3)
	)
	ops[0].OP = "op1"
	ops[1].OP = "op2"
	ops[2].OP = "op3"

	sess, err := mgo.Dial(MGO_HOST)
	assert.Nil(t, err)
	_ = sess.DB(MGO_DB).DropDatabase()
	jobs, err = NewJobsInMgo(cfg)
	assert.Nil(t, err)
	assert.NotNil(t, jobs)

	t.Run("新建Live布控", func(t *testing.T) {
		id, err := jobs.Submit(ctx, uid, utype, "live001", VideoRequest{Ops: ops[0:1]})
		assert.Nil(t, err)
		assert.Equal(t, 12*2, len(id))

		job, err := jobs.Get(ctx, uid, id)
		assert.Nil(t, err)
		assert.Equal(t, StatusWaiting, job.Status)

		assert.Nil(t, jobs.Touch(ctx, &job))
		job, err = jobs.Get(ctx, uid, id)
		assert.Nil(t, err)
		assert.True(t, time.Since(job.UpdatedAt) < time.Millisecond*100)

		err = jobs.Finish(ctx, job)
		assert.Nil(t, err)
		job, err = jobs.Get(ctx, uid, id)
		assert.Nil(t, err)
		assert.Equal(t, StatusFinished, job.Status)
	})

	t.Run("启动Job", func(t *testing.T) {
		defer func() {
			js, _, err := jobs.List(ctx, &uid, nil, nil, nil, nil, nil)
			assert.Nil(t, err)
			for _, job := range js {
				if job.ID != "" {
					assert.Nil(t, jobs.Finish(ctx, job))
				}
			}
		}()

		uid = uid + 1
		_, err = jobs.Submit(ctx, uid, utype, "live002-1", VideoRequest{Ops: ops[0:1]})
		assert.Nil(t, err)
		_, err = jobs.Submit(ctx, uid, utype, "live002-2", VideoRequest{Ops: ops[0:1]})
		assert.Nil(t, err)
		_, err = jobs.Submit(ctx, uid, utype, "live002-3", VideoRequest{Ops: ops[0:1]})
		assert.Nil(t, err)

		// excute one job
		ss, err := jobs.Execute(ctx, 1, map[string]int{"op1": 1})
		assert.Nil(t, err)
		assert.Equal(t, 1, len(ss))
		job, err := jobs.Get(ctx, uid, ss[0].ID)
		assert.Nil(t, err)
		assert.Equal(t, StatusDoing, job.Status)

		// excute two jobs
		ss, err = jobs.Execute(ctx, 2, map[string]int{"op1": 2})
		assert.Nil(t, err)
		assert.Equal(t, 2, len(ss))
		job, err = jobs.Get(ctx, uid, ss[0].ID)
		assert.Nil(t, err)
		assert.Equal(t, StatusDoing, job.Status)
		job, err = jobs.Get(ctx, uid, ss[1].ID)
		assert.Nil(t, err)
		assert.Equal(t, StatusDoing, job.Status)

		// excute more jobs
		ss, err = jobs.Execute(ctx, 1, map[string]int{"op1": 1})
		assert.Nil(t, err)
		assert.Equal(t, 0, len(ss))

		// sleep 100ms to ensure idle timeout
		time.Sleep(100 * time.Millisecond)
		ss, err = jobs.Execute(ctx, 3, map[string]int{"op1": 3})
		assert.Nil(t, err)
		assert.Equal(t, 3, len(ss))

	})

	t.Run("启动Job", func(t *testing.T) {
		defer func() {
			js, _, err := jobs.List(ctx, &uid, nil, nil, nil, nil, nil)
			assert.Nil(t, err)
			for _, job := range js {
				if job.ID != "" {
					assert.Nil(t, jobs.Finish(ctx, job))
				}
			}
		}()

		uid = uid + 1
		id, err := jobs.Submit(ctx, uid, utype, "live003", VideoRequest{Ops: ops[0:1]})
		assert.Nil(t, err)

		js, err := jobs.Execute(ctx, 1, map[string]int{"op1": 1})
		assert.Nil(t, err)
		assert.Equal(t, 1, len(js))

		err = jobs.Cancel(ctx, uid, id)
		assert.Nil(t, err)
		job, err := jobs.Get(ctx, uid, id)
		assert.Nil(t, err)
		assert.Equal(t, StatusCancelling, job.Status)

		err = jobs.Cancelled(ctx, job)
		assert.Nil(t, err)
		job, err = jobs.Get(ctx, uid, id)
		assert.Nil(t, err)
		assert.Equal(t, StatusCancelled, job.Status)
		assert.Equal(t, mgo.ErrNotFound, jobs.Cancelled(ctx, job))

		assert.Equal(t, mgo.ErrNotFound, jobs.Cancel(ctx, uid, id))
	})

	t.Run("查询Job", func(t *testing.T) {
		defer func() {
			js, _, err := jobs.List(ctx, &uid, nil, nil, nil, nil, nil)
			assert.Nil(t, err)
			for _, job := range js {
				if job.ID != "" {
					assert.Nil(t, jobs.Finish(ctx, job))
				}
			}
		}()
		uid = uid + 1
		t1 := time.Now()
		time.Sleep(100 * time.Millisecond)
		_, err := jobs.Submit(ctx, uid, utype, "live003-1", VideoRequest{Ops: ops[0:1]})
		assert.Nil(t, err)
		_, err = jobs.Submit(ctx, uid, utype, "live003-2", VideoRequest{Ops: ops[0:1]})
		assert.Nil(t, err)
		time.Sleep(100 * time.Millisecond)
		t2 := time.Now()
		_, err = jobs.Submit(ctx, uid, utype, "live003-3", VideoRequest{Ops: ops[0:1]})
		assert.Nil(t, err)

		ss, err := jobs.Execute(ctx, 2, map[string]int{"op1": 2})
		assert.Nil(t, err)
		assert.Equal(t, 2, len(ss))
		assert.Nil(t, jobs.Cancel(ctx, uid, ss[0].ID))

		status := StatusWaiting
		s1, _, err := jobs.List(ctx, &uid, &status, nil, nil, nil, nil)
		assert.Nil(t, err)
		assert.Equal(t, 1, len(s1))
		assert.NotEqual(t, ss[0].ID, s1[0].ID)
		assert.NotEqual(t, ss[1].ID, s1[0].ID)
		status = StatusDoing
		s1, _, err = jobs.List(ctx, &uid, &status, nil, nil, nil, nil)
		assert.Nil(t, err)
		assert.Equal(t, 1, len(s1))
		assert.Equal(t, ss[1].ID, s1[0].ID)
		status = StatusCancelling
		s1, _, err = jobs.List(ctx, &uid, &status, nil, nil, nil, nil)
		assert.Nil(t, err)
		assert.Equal(t, 1, len(s1))
		assert.Equal(t, ss[0].ID, s1[0].ID)

		marker := ""
		limit := 2
		s1, marker, err = jobs.List(ctx, &uid, nil, nil, nil, &marker, &limit)
		assert.Nil(t, err)
		assert.Equal(t, 2, len(s1))
		assert.NotEmpty(t, marker)
		s1, marker, err = jobs.List(ctx, &uid, nil, nil, nil, &marker, &limit)
		assert.Nil(t, err)
		assert.Equal(t, 1, len(s1))
		assert.Empty(t, marker)

		s1, marker, err = jobs.List(ctx, &uid, nil, &t1, &t2, nil, nil)
		assert.Nil(t, err)
		assert.Equal(t, 2, len(s1))
		assert.Empty(t, marker)

		s1, marker, err = jobs.List(ctx, &uid, nil, &t1, nil, nil, nil)
		assert.Nil(t, err)
		assert.Equal(t, 3, len(s1))
		assert.Empty(t, marker)

		s1, marker, err = jobs.List(ctx, &uid, nil, nil, &t2, nil, nil)
		assert.Nil(t, err)
		assert.Equal(t, 2, len(s1))
		assert.Empty(t, marker)
	})

	t.Run("多OP任务调度", func(t *testing.T) {
		defer func() {
			js, _, err := jobs.List(ctx, &uid, nil, nil, nil, nil, nil)
			assert.Nil(t, err)
			for _, job := range js {
				if job.ID != "" {
					assert.Nil(t, jobs.Finish(ctx, job))
				}
			}
		}()

		uid = uid + 1
		id1, err := jobs.Submit(ctx, uid, utype, "live004-1", VideoRequest{Ops: ops[0:2]})
		assert.Nil(t, err)
		id2, err := jobs.Submit(ctx, uid, utype, "live004-2", VideoRequest{Ops: ops[0:1]})
		assert.Nil(t, err)
		id3, err := jobs.Submit(ctx, uid, utype, "live004-3", VideoRequest{Ops: ops[1:2]})
		assert.Nil(t, err)
		id4, err := jobs.Submit(ctx, uid, utype, "live004-4", VideoRequest{})
		assert.Nil(t, err)

		// one op, limit=1
		ss, err := jobs.Execute(ctx, 1, map[string]int{"op1": 1})
		assert.Nil(t, err)
		assert.Equal(t, 1, len(ss))
		assert.Equal(t, id2, ss[0].ID)
		// ensure jobb timeout
		time.Sleep(150 * time.Millisecond)

		// two op, limit=2
		ss, err = jobs.Execute(ctx, 2, map[string]int{"op1": 2})
		assert.Nil(t, err)
		assert.Equal(t, 2, len(ss))
		assert.Equal(t, id2, ss[0].ID)
		assert.Equal(t, id4, ss[1].ID)

		// two op, limit=1
		ss, err = jobs.Execute(ctx, 1, map[string]int{"op1": 1, "op2": 1})
		assert.Nil(t, err)
		assert.Equal(t, 1, len(ss))
		assert.Equal(t, id1, ss[0].ID)
		time.Sleep(150 * time.Millisecond)

		// two op, limit=2
		ss, err = jobs.Execute(ctx, 2, map[string]int{"op1": 1, "op2": 1})
		assert.Nil(t, err)
		assert.Equal(t, 2, len(ss))
		assert.Equal(t, id1, ss[0].ID)
		assert.Equal(t, id4, ss[1].ID)
		time.Sleep(150 * time.Millisecond)

		// two op1 + one op2, limit=2
		ss, err = jobs.Execute(ctx, 3, map[string]int{"op1": 1, "op2": 2})
		assert.Nil(t, err)
		assert.Equal(t, 3, len(ss))
		assert.Equal(t, id1, ss[0].ID)
		assert.Equal(t, id4, ss[1].ID)
		assert.Equal(t, id3, ss[2].ID)
		time.Sleep(150 * time.Millisecond)

		// limit=3, enough op
		ss, err = jobs.Execute(ctx, 3, map[string]int{"op1": 2, "op2": 2})
		assert.Nil(t, err)
		assert.Equal(t, 3, len(ss))
		assert.Equal(t, id1, ss[0].ID)
		assert.Equal(t, id2, ss[1].ID)
		assert.Equal(t, id3, ss[2].ID)
		time.Sleep(150 * time.Millisecond)

		ss, err = jobs.Execute(ctx, 4, map[string]int{"op1": 2, "op2": 2})
		assert.Nil(t, err)
		assert.Equal(t, 4, len(ss))
		assert.Equal(t, id1, ss[0].ID)
		assert.Equal(t, id2, ss[1].ID)
		assert.Equal(t, id3, ss[2].ID)
		assert.Equal(t, id4, ss[3].ID)
	})
}

func TestJobsWorker(t *testing.T) {

	var (
		cfg = JobsInMgoConfig{
			Mgo: mgoutil.Config{
				Host: "127.0.0.1:27017",
				DB:   "argus_video_test",
			},
			MgoPoolLimit:   100,
			IdleJobTimeout: 10e8, // 100ms = 10^8 ns
		}
		jobs Jobs
		ops  = make([]struct {
			OP             string   `json:"op"`
			CutHookURL     string   `json:"cut_hook_url"`
			SegmentHookURL string   `json:"segment_hook_url"`
			HookURL        string   `json:"hookURL"`
			Params         OPParams `json:"params"`
		}, 1)
	)
	ops[0].OP = "op1"

	sess, err := mgo.Dial(cfg.Mgo.Host)
	assert.Nil(t, err)
	defer sess.Close()
	assert.Nil(t, sess.DB(cfg.Mgo.DB).DropDatabase())
	jobs, err = NewJobsInMgo(cfg)
	assert.Nil(t, err)
	assert.NotNil(t, jobs)

	w := sync.WaitGroup{}
	var job *Job
	w.Add(1)
	oneJob := func(ctx context.Context, job0 Job) (json.RawMessage, error) {
		job = &job0
		w.Done()
		return nil, nil
	}

	worker := _Worker{
		WorkerConfig:  WorkerConfig{MaxPool: 1, TaskTickerS: 1},
		ops:           map[string]int32{"op1": 1},
		Jobs:          jobs,
		runOneJobFunc: oneJob,
		ch:            make(chan bool, 1),
		count_running: 0,
		closed:        make(chan bool),
	}

	go worker.Run()
	defer worker.Stop()

	_, err = jobs.Submit(context.Background(), 0x01, 0x02, "FOO", VideoRequest{Ops: ops})
	assert.Nil(t, err)
	w.Wait()
	assert.NotNil(t, job)
	assert.Equal(t, "FOO", job.VID)
}
