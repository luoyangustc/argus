package workers

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"net/http"
	"net/url"
	"strconv"
	"sync"

	"github.com/qiniu/http/httputil.v1"
	"github.com/qiniu/xlog.v1"

	job "qiniu.com/argus/bjob/proto"
	"qiniu.com/argus/video"
)

type InferenceVideoTask struct {
	UID    uint32          `json:"uid,omitempty"`
	Utype  uint32          `json:"utype,omitempty"`
	URI    string          `json:"uri"`
	Params json.RawMessage `json:"params,omitempty"`
}

type InferenceVideoConfig struct{}

var _ job.TaskWorker = InferenceVideoWorker{}

type InferenceVideoWorker struct {
	video.Video
	video.OPs
	video.SaverHook
}

func NewInferenceVideoWorker(
	config InferenceImageConfig,
	vide video.Video, ops video.OPs, saverHook video.SaverHook,
) InferenceVideoWorker {
	return InferenceVideoWorker{
		Video:     vide,
		OPs:       ops,
		SaverHook: saverHook,
	}
}

func (w InferenceVideoWorker) Do(ctx context.Context, task job.Task) ([]byte, error) {
	var (
		xl    = xlog.FromContextSafe(ctx)
		_task InferenceVideoTask
	)

	if err := json.Unmarshal(task.Value(ctx), &_task); err != nil {
		xl.Info("parse task error", err)
		return nil, err
	}

	var req video.VideoRequest
	if _task.Params != nil {
		_ = json.Unmarshal(_task.Params, &req)
	}
	req.Data.URI = _task.URI

	var (
		id   = base64.URLEncoding.EncodeToString([]byte(req.Data.URI))
		ends = struct {
			items map[string]struct {
				Labels   []video.ResultLabel   `json:"labels"`
				Segments []video.SegmentResult `json:"segments"`
			}
			sync.Mutex
		}{
			items: make(map[string]struct {
				Labels   []video.ResultLabel   `json:"labels"`
				Segments []video.SegmentResult `json:"segments"`
			}),
		}
		hooks = func(op string) video.EndHook {
			return video.EndHookFunc(
				func(ctx context.Context, rest video.EndResult) error {
					ends.Lock()
					ends.items[op] = rest.Result
					ends.Unlock()
					return nil
				},
			)
		}
		opParams    = make(map[string]video.OPParams)
		saverOPHook video.SaverOPHook
		err         error
	)

	if req.Ops == nil || len(req.Ops) == 0 {
		xl.Warnf("Empty OP Request.")
		return nil, httputil.NewError(http.StatusBadRequest, "empty op")
	}

	for _, op := range req.Ops {
		opParams[op.OP] = op.Params
	}
	ops, ok := w.OPs.Create(ctx, opParams, video.OPEnv{Uid: _task.UID, Utype: _task.Utype})
	if !ok {
		xl.Warnf("Bad OP Request: %#v %#v", w.OPs, opParams)
		return nil, httputil.NewError(http.StatusBadRequest, "bad op")
	}
	xl.Infof("OPS: %#v %#v", w.OPs, ops)

	if w.SaverHook != nil && req.Params.Save != nil {
		saverOPHook, err = w.SaverHook.Get(ctx, _task.UID, id, *req.Params.Save)
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
	}(req.Data.URI, _task.UID)
	if err != nil {
		return nil, err
	}
	err = w.Run(ctx, req, ops, saverOPHook, hooks, nil, nil) // TODO
	if err != nil {
		return nil, err
	}
	bs, _ := json.Marshal(ends.items)
	return bs, nil
}
