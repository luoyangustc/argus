package model

import (
	"context"
	"encoding/json"
	"strconv"

	"qbox.us/errors"

	"github.com/qiniu/xlog.v1"
)

const (
	CONTENT_TYPE   = "Content-Type"
	CONTENT_LENGTH = "Content-Length"
	CT_JSON        = "application/json"
	CT_STREAM      = "application/octet-stream"
)

// IsJsonContent ...
func IsJsonContent(contentType string) bool {
	switch contentType {
	case CT_JSON, "application/json; charset=UTF-8", "application/json; charset=utf-8":
		return true
	default:
		return false
	}
}

////////////////////////////////////////////////////////////////////////////////

type TaskReq interface {
	GetCmd() string
	GetVersion() *string
	Marshal() []byte
}

// STRING string
type STRING string

func (s STRING) String() string { return string(s) }

// GoString ...
func (s STRING) GoString() string {
	if len(s) > 256 {
		return string(s)[:253] + "..."
	}
	return string(s)
}

// BYTES []byte
type BYTES []byte

func (bs BYTES) Bytes() []byte    { return []byte(bs) }
func (bs BYTES) GoString() string { return strconv.Itoa(len(bs)) }

type Resource struct {
	URI       STRING      `json:"uri"`
	Attribute interface{} `json:"attribute,omitempty"`
}

type ResourceInner struct {
	URI       interface{} `json:"uri"`
	Attribute interface{} `json:"attribute,omitempty"`
}

type EvalResponse struct {
	Code    int         `json:"code"`
	Message string      `json:"message"`
	Result  interface{} `json:"result,omitempty"`
}

var _ TaskReq = EvalRequest{}

type EvalRequest struct {
	OP      string      `json:"op,omitempty"`
	Cmd     string      `json:"-"`
	Version *string     `json:"-"`
	Data    Resource    `json:"data"`
	Params  interface{} `json:"params,omitempty"`
}

func (req EvalRequest) GetCmd() string      { return req.Cmd }
func (req EvalRequest) GetVersion() *string { return req.Version }
func (req EvalRequest) Marshal() []byte {
	bs, _ := json.Marshal(req)
	return bs
}

type EvalRequestInner struct {
	OP      string        `json:"op,omitempty"`
	Cmd     string        `json:"-"`
	Version *string       `json:"-"`
	Data    ResourceInner `json:"data"`
	Params  interface{}   `json:"params,omitempty"`
}

func ToEvalRequestInner(req EvalRequest) EvalRequestInner {
	return EvalRequestInner{
		OP:      req.OP,
		Cmd:     req.Cmd,
		Version: req.Version,
		Data: ResourceInner{
			URI:       req.Data.URI,
			Attribute: req.Data.Attribute,
		},
		Params: req.Params,
	}
}

var _ TaskReq = GroupEvalRequest{}

type GroupEvalRequest struct {
	OP      string      `json:"op,omitempty"`
	Cmd     string      `json:"-"`
	Version *string     `json:"-"`
	Data    []Resource  `json:"data"`
	Params  interface{} `json:"params,omitempty"`
}

func (req GroupEvalRequest) GetCmd() string      { return req.Cmd }
func (req GroupEvalRequest) GetVersion() *string { return req.Version }
func (req GroupEvalRequest) Marshal() []byte {
	bs, _ := json.Marshal(req)
	return bs
}

type GroupEvalRequestInner struct {
	OP      string          `json:"op,omitempty"`
	Cmd     string          `json:"-"`
	Version *string         `json:"-"`
	Data    []ResourceInner `json:"data"`
	Params  interface{}     `json:"params,omitempty"`
}

func ToGroupEvalRequestInner(req GroupEvalRequest) GroupEvalRequestInner {
	datas := make([]ResourceInner, 0, len(req.Data))
	for _, data := range req.Data {
		datas = append(datas, ResourceInner{URI: data.URI, Attribute: data.Attribute})
	}
	return GroupEvalRequestInner{
		OP:      req.OP,
		Cmd:     req.Cmd,
		Version: req.Version,
		Data:    datas,
		Params:  req.Params,
	}
}

//----------------------------------------------------------------------------//

var (
	ErrParseTaskRequest error = errors.New("parse task request failed")
)

// 暂时靠多一次Marshal、Unmarshal

func UnmarshalTaskRequest(ctx context.Context, data []byte) (TaskReq, error) {
	var xl = xlog.FromContextSafe(ctx)
	{
		var value EvalRequest
		err := json.Unmarshal(data, &value)
		if err != nil {
			xl.Warnf("unmarshal request as EvalRequest. %v", err)
		}
		if err == nil && len(value.Data.URI) > 0 {
			return value, nil
		}
	}
	var value GroupEvalRequest
	err := json.Unmarshal(data, &value)
	xl.Infof("unmarshal request as GroupEvalRequest. %v", err)
	if err != nil || value.Data == nil || len(value.Data) == 0 {
		return nil, ErrParseTaskRequest
	}
	return value, err
}

type _TaskRequest struct {
	OP     string      `json:"op,omitempty"`
	Data   interface{} `json:"data"`
	Params interface{} `json:"params,omitempty"`
}

func UnmarshalBatchTaskRequest(ctx context.Context, data []byte) ([]TaskReq, error) {
	var tasks1 []_TaskRequest
	if err := json.Unmarshal(data, &tasks1); err != nil {
		return nil, ErrParseTaskRequest
	}
	var tasks2 []TaskReq = make([]TaskReq, 0, len(tasks1))
	for _, task := range tasks1 {
		bs, _ := json.Marshal(task)
		v, err := UnmarshalTaskRequest(ctx, bs)
		if err != nil {
			return nil, err
		}
		tasks2 = append(tasks2, v)
	}
	return tasks2, nil
}
