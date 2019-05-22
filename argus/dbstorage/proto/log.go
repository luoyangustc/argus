package proto

import (
	"fmt"
)

type ErrorCode int

type Log struct {
	TaskId   TaskId     `bson:"task_id"`
	Uid      uint32     `bson:"uid"`
	Process  []int      `bson:"process"`
	Hash     []string   `bson:"hash"`
	Err      []ErrorLog `bson:"error"`
	ErrCount int        `bson:"error_count"`
}

type ErrorLog struct {
	Uri     string    `json:"uri"`
	Code    ErrorCode `json:"code"`
	Message string    `json:"message"`
}

func (e ErrorLog) String() string {
	return fmt.Sprintf("%s : %d : %s", e.Uri, e.Code, e.Message)
}
