package censor

import (
	pimage "qiniu.com/argus/service/service/image"
)

const (
	DEFAULT_INTERVAL int64  = 5000
	PULP             string = "pulp"
	TERROR           string = "terror"
	POLITICIAN       string = "politician"
)

type CutParam struct {
	Mode          int   `json:"mode"`           // 模式，0:间隔帧，1：关键帧
	IntervalMsecs int64 `json:"interval_msecs"` // 单位毫秒
}

type SaveParam struct {
	Save bool `json:"save"`
}

type VideoCensorReq struct {
	Data struct {
		URI string `json:"uri"`
	} `json:"data"`
	Params struct {
		Scenes   []string  `json:"scenes"`
		CutParam *CutParam `json:"cut_param,omitempty"`
		Saver    SaveParam `json:"saver,omitempty"`
	} `json:"params"`
}

type VideoCensorResp struct {
	Code    int           `json:"code"`
	Message string        `json:"message"`
	Result  *CensorResult `json:"result,omitempty"`
}

type CensorResult struct {
	Suggestion pimage.Suggestion       `json:"suggestion"` // 审核结论
	Scenes     map[string]*SceneResult `json:"scenes,omitempty"`
}

type SceneResult struct {
	Suggestion pimage.Suggestion `json:"suggestion"`     // 审核结论-单场景
	Cuts       []CutResult       `json:"cuts,omitempty"` // 帧结果
}

type CutResult struct {
	Suggestion pimage.Suggestion `json:"suggestion"`        // 审核结论-单场景-单帧
	Offset     int64             `json:"offset"`            // 帧时间
	Uri        string            `json:"uri,omitempty"`     // 帧存储地址
	Details    []pimage.Detail   `json:"details,omitempty"` // 标签明细
}
