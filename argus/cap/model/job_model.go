package model

import (
	"qiniu.com/argus/cap/dao"
	"qiniu.com/argus/cap/enums"
)

// JobQueryReq
type JobQueryReq struct {
	CmdArgs []string // JobID
}

// JobQueryResp
type JobQueryResp struct {
	JobModel
}

// JobCreateReq
type JobCreateReq struct {
	JobID     string         `json:"jobId"`
	JobType   enums.JobType  `json:"jobType"`
	LabelMode string         `json:"labelMode"`
	MimeType  enums.MimeType `json:"mimeType"` //image || video
	Uid       uint32         `json:"uid"`      //发起审核请求的用户id
}

// JobCreateResp
type JobCreateResp struct {
	JobID string `json:"jobId"`
}

// JobTaskReq
type JobTaskReq struct {
	CmdArgs []string // JobID
	Tasks   []struct {
		ID     string      `json:"id"`
		URI    string      `json:"uri"` // HTTP|QINIU
		Labels []LabelInfo `json:"label"`
	} `json:"tasks"`
}

//Result 相关的
type GetResultReq struct {
	CmdArgs []string // JobID

	Limit  int    `json:"limit,omitempty"`
	Marker string `json:"marker,omitempty"` // just for batch job
}

type JobCheckResultReq struct {
	CmdArgs []string // JobID
}

type JobCheckResultResp struct {
	Finish bool `json:"bFinish"`
}

//==================================================================================

type JobModel struct {
	JobID      string              `json:"jobId"`
	JobType    enums.JobType       `json:"jobType"`
	LabelMode  string              `json:"labelMode"`
	MimeType   enums.MimeType      `json:"mimeType"` //image || video
	CreateTime string              `json:"createTime"`
	Status     enums.JobStatusType `json:"status"`
	Uid        uint32              `json:"uid"` //发起审核请求的用户id
}

func FromJobInMgo(jobInMgo *dao.JobInMgo) *JobModel {
	return &JobModel{
		JobID:      jobInMgo.JobID,
		JobType:    enums.JobType(jobInMgo.JobType),
		LabelMode:  jobInMgo.LabelMode,
		MimeType:   enums.MimeType(jobInMgo.MimeType),
		CreateTime: jobInMgo.CreateTime.Format("20060102T150405"),
		Status:     enums.JobStatusType(jobInMgo.Status),
		Uid:        jobInMgo.Uid,
	}
}

func ToJobInMgo(jobModel *JobModel) *dao.JobInMgo {
	return &dao.JobInMgo{
		JobID:     jobModel.JobID,
		JobType:   string(jobModel.JobType),
		LabelMode: jobModel.LabelMode,
		MimeType:  string(jobModel.MimeType),
		Uid:       jobModel.Uid,
	}
}
