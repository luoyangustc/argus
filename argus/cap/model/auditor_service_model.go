package model

//基本定义，用于其他结构之中
type LabelTitle struct {
	Title    string `json:"title"`
	Desc     string `json:"desc"`
	Selected bool   `json:"selected"`
}

//用于与labelx交互
type TaskResult struct {
	TaskID string      `json:"taskId"`
	URI    string      `json:"url"`
	Labels []LabelInfo `json:"label"` //LabelData | LabelPoliticianData
}

//=================================Auditor=============================
// GetAuditorAttr_
type GetAuditorAttrReq struct {
	CmdArgs []string //auditorId
}

type GetAuditorAttrResp struct {
	AuditorID     string `json:"auditorId"`
	Valid         string `json:"valid"` // 注销 逻辑删除
	RealTimeLevel string `json:"realTimeLevel"`
	CurLabel      string `json:"curLabel"`
}

//=================================Task================================
// GetRealtimeTask_
type GetRealtimeTaskReq struct {
	CmdArgs []string //auditorId
}

type GetRealtimeTaskResp struct {
	AuditorID   string                  `json:"auditorId"`
	PID         string                  `json:"pid"`
	Type        string                  `json:"type"`     // image | video| live
	TaskType    []string                `json:"taskType"` // 如果有politician，就都改成 detect.xxx，没有就保持 classify.pulp...
	Labels      map[string][]LabelTitle `json:"label"`
	CallbackURI string                  `json:"callbackUri"`
	Mode        string                  `json:"mode"` // realtime | batch
	IndexData   []TaskResult            `json:"indexData"`
	ExpiryTime  string                  `json:"expiryTime"`
}

//============================
// PostCancelRealtimeTask
type PostCancelRealtimeTaskReq struct {
	AuditorID string   `json:"auditorId"`
	PID       string   `json:"pid"`
	TaskIds   []string `json:"taskids"`
}

type PostCancelRealtimeTaskResp struct {
	Code int    `json:"code"`
	Msg  string `json:"msg"`
}

// PostResult
type PostResultReq struct {
	AuditorID string       `json:"auditorId"`
	PID       string       `json:"pid"`
	Success   bool         `json:"success"`
	Result    []TaskResult `json:"result"`
}

type PostResultResp struct {
	Code int    `json:"code"`
	Msg  string `json:"msg"`
}
