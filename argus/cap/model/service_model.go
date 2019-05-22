package model

////////////////////////////////////////////////////////////////

type AuditorReq struct {
	Data struct {
		ID string `json:"auditorId"` //用户ID
	} `json:"data"`
}

////////////////////////////////////////////////////////////////

type PostAuditorResultReq struct {
	PackageID string    `json:"pid"`
	AuditorID string    `json:"auditorId"`
	Result    TaskModel `json:"result"`
}

type GetSandsReq struct {
	LabelType string `json:"label_type"`
	Num       int    `json:"num"`
}

type GetSandsResp struct {
	Data []*TaskModel `json:"data"`
}

type SandTask struct {
	URI   string      `json:"uri"` // HTTP|QINIU
	Label []LabelInfo `json:"label"`
}

// AddSandRequest AddSandRequest
type AddSandRequest struct {
	Tasks []SandTask `json:"tasks"`
}

// AddSandRequest AddSandRequest
type AddSandFilesRequest struct {
	Files []string `json:"files"`
}
