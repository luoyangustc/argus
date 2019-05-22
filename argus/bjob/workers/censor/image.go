package censor

import (
	"context"
	"encoding/json"
	"time"

	httputil "github.com/qiniu/http/httputil.v1"
	"github.com/qiniu/xlog.v1"

	job "qiniu.com/argus/bjob/proto"
	"qiniu.com/argus/censor"
	"qiniu.com/argus/censor/biz"
)

var _ job.TaskWorker = ImageWorker{}

type ImageParams struct {
	Scenes []biz.Scene `json:"scenes,omitempty"`
	Params struct {
		Scenes map[biz.Scene]json.RawMessage `json:"scenes"`
	} `json:"params,omitempty"`
}

type ImageConfig struct {
	Host          string        `json:"host"`
	TimeoutSecond time.Duration `json:"timeout_second"`
}

type ImageWorker struct {
	censor.NewImageCensorClient
}

func NewImageWorker(conf ImageConfig) ImageWorker {
	return ImageWorker{
		NewImageCensorClient: censor.NewImageCensorHTTPClient(
			conf.Host, time.Second*conf.TimeoutSecond),
	}
}

func (w ImageWorker) Do(ctx context.Context, task job.Task) ([]byte, error) {
	var (
		xl     = xlog.FromContextSafe(ctx)
		_task  Task
		result []byte
	)

	if err := json.Unmarshal(task.Value(ctx), &_task); err != nil {
		xl.Info("parse task error", err)
		return nil, err
	}
	var params = ImageParams{}
	_ = json.Unmarshal(_task.Params, &params)
	if len(params.Scenes) == 0 {
		params.Scenes = biz.DefaultScenes
	}

	cli := w.NewImageCensorClient(_task.UID, _task.Utype)
	ret := censor.ImageRecognition(ctx, _task.URI, cli, params.Scenes, nil)
	if ret.Code >= 300 {
		return nil, httputil.NewError(ret.Code, ret.Message)
	}
	result, _ = json.Marshal(ret)
	return result, nil
}
