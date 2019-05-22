package embedEval

import (
	"context"
	"log"
	"sync"

	"qiniu.com/argus/atserving/model"
	"qiniu.com/argus/serving_eval"
)

// Worker ...
type Worker interface {
	Eval(context.Context, []model.EvalRequest) ([]model.EvalResponse, error)
}

// Config ...
type Config struct {
	Workspace string `json:"workspace"`

	ModelParams interface{} `json:"model_params"`

	BatchSize  int    `json:"batch_size,omitempty"`
	ImageWidth int    `json:"image_width"`
	UseDevice  string `json:"use_device"`

	TarFiles     map[string]string      `json:"tar_files"`
	CustomFiles  map[string]string      `json:"custom_files,omitempty"`
	CustomValues map[string]interface{} `json:"custom_values,omitempty"`
}

type worker struct {
	Config
	eval.Handler

	*sync.Mutex
}

// NewWorker ...
func NewWorker(ctx context.Context, config Config) (Worker, error) {
	if config.BatchSize <= 0 {
		config.BatchSize = 1
	}

	var evalConfig = &eval.EvalConfig{
		App: "",

		Workspace:   config.Workspace,
		ModelParams: config.ModelParams,
		BatchSize:   config.BatchSize,
		ImageWidth:  config.ImageWidth,
		UseDevice:   config.UseDevice,

		TarFiles:     config.TarFiles,
		CustomFiles:  config.CustomFiles,
		CustomValues: config.CustomValues,
	}
	core, err := eval.NewEvalNetLocal(ctx, evalConfig)
	if err != nil {
		log.Printf("Init eval net failed, err: %s", err)
		return nil, err
	}
	handler := eval.NewHandler(core, config.Workspace)

	return &worker{
		Config:  config,
		Handler: handler,
		Mutex:   new(sync.Mutex),
	}, nil
}

func (w *worker) Eval(
	ctx context.Context, reqs []model.EvalRequest,
) ([]model.EvalResponse, error) {

	w.Lock()
	defer w.Unlock()

	var (
		i, n  = 0, len(reqs)
		resps = make([]model.EvalResponse, n)
	)

	for {
		if i >= n {
			break
		}
		var j = i + w.BatchSize
		if j > n {
			j = n
		}
		var _reqs = make([]model.EvalRequestInner, 0)
		for _, req := range reqs[i:j] {
			_reqs = append(_reqs, model.ToEvalRequestInner(req))
		}
		_resps, err := w.Handler.Eval(ctx, _reqs)
		if err != nil {
			return nil, err
		}
		for k := i; k < j; k++ {
			resps[k] = model.EvalResponse{
				Code:    _resps[k-i].Code,
				Message: _resps[k-i].Message,
				Result:  _resps[k-i].Result,
			}
		}
		i = j
	}

	return resps, nil
}
