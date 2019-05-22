package proto

// ReqData ...
type ReqData struct {
	URI string `json:"uri"`
}

// ReqCutPara ...
type ReqCutPara struct {
	Mode          int   `json:"mode,omitempty"`
	IntervalMsecs int32 `json:"interval_msecs"` // 单位毫秒, [1000, 60000]
}

// ReqParams ...
type ReqParams struct {
	Scenes   []string    `json:"scenes"`
	CutParam *ReqCutPara `json:"cut_param,omitempty"`
}

// VReqV3 ...
type VReqV3 struct {
	Data   ReqData   `json:"data"`
	Params ReqParams `json:"params"`
}

// VRespJob ...
type VRespJob struct {
	Job string `json:"job"`
}

// VRespJobResult ...
type VRespJobResult struct {
	ID      string  `json:"id"`
	Vid     string  `json:"vid"`
	Request VReqV3  `json:"request"`
	Status  string  `json:"status"`
	Result  VRespV3 `json:"result"`
	Error   string  `json:"error"`
}

// VRespV3 ...
type VRespV3 struct {
	Code    int           `json:"code"`
	Message string        `json:"message"`
	Result  VRespV3Result `json:"result"`
}

// VRespV3Result ...
type VRespV3Result struct {
	Suggestion string `json:"suggestion"`
	Scenes     struct {
		Pulp       ScenesResult `json:"pulp,omitempty"`
		Politician ScenesResult `json:"politician,omitempty"`
		Terror     ScenesResult `json:"terror,omitempty"`
	} `json:"scenes,omitempty"`
}

// ScenesResult ...
type ScenesResult struct {
	Suggestion string       `json:"suggestion"`
	Cuts       []*CutResult `json:"cuts,omitempty"`
}

// CutResult ...
type CutResult struct {
	Suggestion string       `json:"suggestion"`
	Offset     int32        `json:"offset"`
	URI        string       `json:"uri,omitempty"`
	Details    []*CutDetail `json:"details,omitempty"`
}

// CutDetail ...
type CutDetail struct {
	Suggestion string  `json:"suggestion"`         // 审核结论-单标签
	Label      string  `json:"label"`              // 标签
	Group      string  `json:"group,omitempty"`    // 分组
	Score      float32 `json:"score"`              // 置信度
	Comments   string  `json:"comments,omitempty"` // 提示描述
	Detections []struct {
		Pts   [][2]int `json:"pts"`   // 坐标
		Score float32  `json:"score"` //检测框置信度
	} `json:"detections,omitempty"` // 检测框
}

// VRespJobStatus ...
type VRespJobStatus struct {
	ID        string `json:"id"`
	Status    string `json:"status"`
	CreatedAt string `json:"created_at"`
	UpdatedAt string `json:"updated_at"`
}

// NewReq ...
func NewCensorVideoReq(uri string, scenes []string, interval int32) *VReqV3 {
	request := &VReqV3{
		Data: ReqData{
			URI: uri,
		},
		Params: ReqParams{
			Scenes: scenes,
			CutParam: &ReqCutPara{
				IntervalMsecs: interval,
			},
		},
	}

	return request
}
