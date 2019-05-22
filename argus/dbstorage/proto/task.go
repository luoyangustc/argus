package proto

import (
	"errors"
	"strconv"
)

type GroupName string
type TaskStatus string
type TaskError string
type TaskId string

const (
	MIN_FACE_SIZE = 50
	MODE_SINGLE   = "SINGLE"
	MODE_LARGEST  = "LARGEST"
)
const (
	CREATED   TaskStatus = "Created"
	PENDING   TaskStatus = "Pending"
	RUNNING   TaskStatus = "Running"
	STOPPING  TaskStatus = "Stopping"
	STOPPED   TaskStatus = "Stopped"
	COMPLETED TaskStatus = "Completed"
)

type Task struct {
	TaskId       TaskId     `bson:"task_id"`
	Uid          uint32     `bson:"uid"`
	GroupName    GroupName  `bson:"group_name"`
	Config       TaskConfig `bson:"task_config"`
	TotalCount   int        `bson:"total_count"`
	HandledCount int        `bson:"handled_count"`
	Status       TaskStatus `bson:"status"`
	FileName     string     `bson:"file_name"`
	FileExt      string     `bson:"file_ext"`
	LastError    TaskError  `bson:"last_error"`
}

type TaskConfig struct {
	RejectBadFace bool   `json:"reject_bad_face" bson:"reject_bad_face"`
	Mode          string `json:"mode" bson:"mode"`
}

func GetValidBool(s string) (bool, error) {
	if s == "" {
		return false, nil
	}
	filter, err := strconv.ParseBool(s)
	if err != nil {
		return false, errors.New("not a bool")
	}
	return filter, nil
}

func GetValidMode(s string) (string, error) {
	switch s {
	case "", MODE_SINGLE:
		return MODE_SINGLE, nil
	case MODE_LARGEST:
		return MODE_LARGEST, nil
	default:
		return "", errors.New("invalid mode")
	}
}
