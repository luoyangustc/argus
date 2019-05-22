package vframe

import (
	"bytes"
	"context"
	"encoding/base64"
	"io"

	xlog "github.com/qiniu/xlog.v1"
	URI "qiniu.com/argus/com/uri"
)

type vframe_base64 struct {
	Vframe Vframe
}

type _JobBase64 struct {
	ch   chan CutResponse
	_Job Job
}

func NewVframeBase64(vf Vframe) Vframe {
	return vframe_base64{Vframe: vf}
}

func (f vframe_base64) Run(ctx context.Context, req VframeRequest) (Job, error) {
	origJob, err := f.Vframe.Run(ctx, req)
	if err != nil {
		return nil, err
	}
	job := &_JobBase64{
		ch:   make(chan CutResponse),
		_Job: origJob,
	}

	go job.run(ctx)
	return job, nil
}

func (j *_JobBase64) run(ctx context.Context) {
	var (
		handler = newCutHandler()
		xl      = xlog.FromContextSafe(ctx)
	)

	defer func() {
		if jc, ok := j._Job.(JobClean); ok {
			jc.Clean()
		}
	}()

	for cut := range j._Job.Cuts() {
		cut.Result.Cut.URI = func(uri string) string {
			resp, err := handler.Get(ctx, URI.Request{URI: uri})
			if err != nil { // TODO
				xl.Warnf("get uri failed. %s %v", uri, err)
				return ""
			}
			defer func() {
				resp.Body.Close()
				_ = handler.Del(ctx, URI.Request{URI: uri})
			}()

			buf := bytes.NewBuffer(nil)
			base64W := base64.NewEncoder(base64.StdEncoding, buf)
			_, _ = io.Copy(base64W, resp.Body)
			base64W.Close()
			return "data:application/octet-stream;base64," + buf.String()
		}(cut.Result.Cut.URI)

		if cut.Result.Cut.URI == "" {
			continue
		}

		xl.Debugf("_JobBase64 uri: %v", len(cut.Result.Cut.URI))
		j.ch <- cut
	}
	xl.Info("_JobBase64: close channel")
	close(j.ch)

}

func (j *_JobBase64) Cuts() <-chan CutResponse { return j.ch }
func (j *_JobBase64) Stop()                    { j._Job.Stop() }
func (j *_JobBase64) Error() error             { return j._Job.Error() }
