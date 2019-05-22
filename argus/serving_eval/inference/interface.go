package inference

import (
	"context"
	"net/http"

	"qiniu.com/argus/atserving/model"
)

type InitParams struct {
	App string `json:"app"`
}

type CreateParams struct {
	App           string `json:"app"`
	Workspace     string `json:"workspace"`
	UseDevice     string `json:"use_device"`
	BatchSize     int    `json:"batch_size,omitempty"`
	MaxConcurrent int    `json:"max_concurrent,omitempty"`

	ModelFiles   map[string]string      `json:"model_files"`
	ModelParams  interface{}            `json:"model_params"`
	CustomFiles  map[string]string      `json:"custom_files,omitempty"`
	CustomParams map[string]interface{} `json:"custom_params,omitempty"`
}

type Creator interface {
	Create(context.Context, *CreateParams) (Instance, error)
}

type Response struct {
	Code    int         `json:"code"`
	Message string      `json:"message"`
	Header  http.Header `json:"header,omitempty"`
	Result  interface{} `json:"result,omitempty"`
	Body    []byte      `json:"body,omitempty"`
}

type Instance interface {
	Preprocess(context.Context, model.EvalRequestInner) (model.EvalRequestInner, error)
	PreprocessGroup(context.Context, model.GroupEvalRequestInner) (model.GroupEvalRequestInner, error)

	Inference(context.Context, []model.EvalRequestInner) ([]Response, error)
	InferenceGroup(context.Context, []model.GroupEvalRequestInner) ([]Response, error)
}

////////////////////////////////////////////////////////////////////////////////

var (
	_AllowedCode = []int{
		0, 200, // 正确返回
		400, // 输入资源、参数错误
		424, // 资源文件获取失败
		500, // 内部错误
		599, // 未知错误
	}
)

func inAllowedCode(code int) bool {
	for _, _code := range _AllowedCode {
		if code == _code {
			return true
		}
	}
	return false
}

func foramtCodeMessage(code int, message string) (int, string) {
	if inAllowedCode(code) {
		return code, message
	}
	return http.StatusInternalServerError, message
}
