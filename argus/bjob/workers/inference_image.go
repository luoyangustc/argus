package workers

import (
	"bytes"
	"context"
	"encoding/json"
	"io/ioutil"
	"text/template"
	"time"

	"github.com/qiniu/rpc.v3"
	"github.com/qiniu/xlog.v1"

	ahttp "qiniu.com/argus/argus/com/http"
	job "qiniu.com/argus/bjob/proto"
)

type InferenceImageTask struct {
	UID    uint32          `json:"uid,omitempty"`
	Utype  uint32          `json:"utype,omitempty"`
	URI    string          `json:"uri"`
	Params json.RawMessage `json:"params,omitempty"`
}

type InferenceImageConfig struct {
	RequestTemplate string
	URL             string
}

var _ job.TaskWorker = InferenceImageWorker{}

type InferenceImageWorker struct {
	InferenceImageConfig
	InferenceImageRequestFormat
}

func NewInferenceImageWorker(config InferenceImageConfig) InferenceImageWorker {
	return InferenceImageWorker{
		InferenceImageConfig:        config,
		InferenceImageRequestFormat: newInferenceImageRequestFormat(config.RequestTemplate),
	}
}

func (w InferenceImageWorker) Do(ctx context.Context, task job.Task) ([]byte, error) {
	var (
		xl     = xlog.FromContextSafe(ctx)
		_task  InferenceImageTask
		result []byte
	)

	if err := json.Unmarshal(task.Value(ctx), &_task); err != nil {
		xl.Info("parse task error", err)
		return nil, err
	}

	buf := w.Format(_task)
	xl.Infof("ImageWorker url, %s", w.InferenceImageConfig.URL)
	xl.Infof("ImageWorker req, %s", buf.String())

	var (
		client = NewRPCClient(_task.UID, _task.Utype, time.Second*60) // TODO
		f      = func(ctx context.Context) error {
			resp, err := client.DoRequestWith(ctx,
				"POST", w.InferenceImageConfig.URL, "application/json",
				buf, buf.Len())
			if err != nil {
				return err
			}
			defer resp.Body.Close()
			if resp.StatusCode/100 == 2 {
				bs, _ := ioutil.ReadAll(resp.Body)
				result = bs
			} else {
				err = rpc.ResponseError(resp)
			}
			return err
		}
	)
	err := ahttp.CallWithRetry(ctx, []int{530}, []func(context.Context) error{f, f})
	if err != nil {
		xl.Warnf("Call Failed. %s %v", _task.URI, err)
	}
	return result, err
}

type InferenceImageRequestFormat struct {
	*template.Template
}

func newInferenceImageRequestFormat(tmpl string) InferenceImageRequestFormat {
	if tmpl == "" {
		tmpl = `{"data":{"uri":"{{.URI}}"}{{if .Params}},"params":{{.Params}}{{end}}}`
	}
	return InferenceImageRequestFormat{
		Template: template.Must(template.New("request").Parse(tmpl)),
	}
}

func (format InferenceImageRequestFormat) Format(task InferenceImageTask) *bytes.Buffer {
	var buf = bytes.NewBuffer(nil)
	_ = format.Template.Execute(buf,
		struct {
			URI    string
			Params string
		}{URI: task.URI, Params: string(task.Params)})
	return buf
}

//----------------------------------------------------------------------------//

// NewRPCClient FOR HACK
var NewRPCClient = func(uid, utype uint32, timeout time.Duration) *rpc.Client {
	return ahttp.NewQiniuStubRPCClient(uid, utype, timeout)
}
