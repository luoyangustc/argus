package censor

import (
	"context"
	"encoding/json"

	job "qiniu.com/argus/bjob/proto"
)

var _ job.Task = &_Task{}

type _Task struct {
	_Value []byte
}

func NewTask(value []byte) *_Task             { return &_Task{_Value: value} }
func (t *_Task) Value(context.Context) []byte { return t._Value }

type Task struct {
	UID      uint32          `json:"uid,omitempty"`
	Utype    uint32          `json:"utype,omitempty"`
	URI      string          `json:"uri"`
	Mimetype string          `json:"mimetype,omitempty"`
	Params   json.RawMessage `json:"params,omitempty"`
}
