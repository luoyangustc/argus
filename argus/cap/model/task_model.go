package model

import (
	"encoding/json"

	"qiniu.com/argus/cap/dao"
	"qiniu.com/argus/cap/enums"
)

//========================================================================================
type LabelData struct {
	Class string  `json:"class"`
	Score float32 `json:"score"`
}

type LabelPoliticianData struct {
	Class string `json:"class"`
	Faces []struct {
		BoundingBox struct {
			Pts   [][2]int `json:"pts"`
			Score float32  `json:"score"`
		} `json:"bounding_box"`
		Faces []struct {
			ID     string  `json:"id,omitempty"`
			Name   string  `json:"name,omitempty"`
			Score  float32 `json:"score"`
			Group  string  `json:"group,omitempty"`
			Sample *struct {
				URL string   `json:"url"`
				Pts [][2]int `json:"pts"`
			} `json:"sample,omitempty"`
		} `json:"faces,omitempty"`
	} `json:"faces"`
}

//Task LabelInfo
type LabelInfo struct {
	Name string              `json:"name"`  // pulp | terror | policitian
	Type enums.TaskLabelType `json:"type"`  // classification | detection，要与外面TaskType保持一致
	Data []interface{}       `json:"data" ` //LabelData | LabelPoliticianData
}

// TaskInfo
type TaskModel struct {
	TaskID    string      `json:"taskId"`
	JobID     string      `json:"jobId"`
	URI       string      `json:"url"`
	Labels    []LabelInfo `json:"label"`
	Status    string      `json:"status,omitempty"`
	AuditorID string      `json:"auditorId,omitempty"` //最终完成该标注任务的标注人员Id
	Result    []LabelInfo `json:"result"`
}

func FromTaskInMgo(taskInMgo *dao.TaskInMgo) *TaskModel {
	taskmodel := TaskModel{
		TaskID:    taskInMgo.TaskID,
		JobID:     taskInMgo.JobID,
		URI:       taskInMgo.URI,
		Status:    taskInMgo.Status,
		AuditorID: taskInMgo.AuditorID,
	}

	var labels []LabelInfo
	err := json.Unmarshal(taskInMgo.Labels, &labels)
	if err == nil {
		taskmodel.Labels = labels
	}

	var result []LabelInfo
	err = json.Unmarshal(taskInMgo.Result, &result)
	if err == nil {
		taskmodel.Result = result
	}

	return &taskmodel
}

func ToTaskInMgo(taskModel *TaskModel) *dao.TaskInMgo {
	taskInMgo := dao.TaskInMgo{
		TaskID:    taskModel.TaskID,
		JobID:     taskModel.JobID,
		URI:       taskModel.URI,
		Status:    taskModel.Status,
		AuditorID: taskModel.AuditorID,
	}

	labels, err := json.Marshal(taskModel.Labels)
	if err == nil {
		taskInMgo.Labels = labels
	}

	result, err := json.Marshal(taskModel.Result)
	if err == nil {
		taskInMgo.Result = result
	}

	return &taskInMgo
}

func ToTaskResult(tModel *TaskModel) *TaskResult {
	return &TaskResult{
		TaskID: tModel.TaskID,
		URI:    tModel.URI,
		Labels: tModel.Labels,
	}
}
