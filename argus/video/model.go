package video

import (
	"encoding/json"

	"qiniu.com/argus/video/vframe"
)

////////////////////////////////////////////////////////////////////////////////

type VideoRequest struct {
	CmdArgs []string `json:"-"`
	Data    struct {
		URI       string `json:"uri"`
		Attribute struct {
			ID   string          `json:"id"`
			Meta json.RawMessage `json:"meta,omitempty"`
		} `json:"attribute"`
	}
	Params struct {
		Async         bool                 `json:"async"`
		SegmentParams *SegmentParams       `json:"segment"`
		Vframe        *vframe.VframeParams `json:"vframe"`
		Live          *struct {
			Timeout    float64 `json:"timeout"`
			Downstream string  `json:"downstream"`
		} `json:"live"`
		Save    *json.RawMessage `json:"save,omitempty"`
		HookURL string           `json:"hookURL"`
	} `json:"params"`
	Ops []struct {
		OP             string   `json:"op"`
		CutHookURL     string   `json:"cut_hook_url"`
		SegmentHookURL string   `json:"segment_hook_url"`
		HookURL        string   `json:"hookURL"`
		Params         OPParams `json:"params"`
	} `json:"ops"`
}

type SegmentParams struct {
	Mode     int `json:"mode"`
	Interval int `json:"interval"`
}

type OPParams struct {
	Labels []struct {
		Name   string  `json:"label"`
		Select int     `json:"select"` // 0x01:INGORE; 0x02:ONLY
		Score  float32 `json:"score"`
	} `json:"labels"`
	Terminate struct {
		Mode   int            `json:"mode"` // 0x01:cut; 0x02:segment
		Labels map[string]int `json:"labels"`
	} `json:"terminate"`
	IgnoreEmptyLabels bool        `json:"ignore_empty_labels"`
	Other             interface{} `json:"other"`
}

////////////////////////////////////////////////////////////////////////////////

type ResultLabel struct {
	Name  string  `json:"label"`
	Score float32 `json:"score"`
}

type CutResult struct {
	Offset int64       `json:"offset"`
	URI    string      `json:"uri,omitempty"`
	Result interface{} `json:"result"`
}

type ClipResult struct {
	OffsetBegin int64       `json:"offset_begin"`
	OffsetEnd   int64       `json:"offset_end"`
	Result      interface{} `json:"result"`
}

type SegmentResult struct {
	OP          string        `json:"op,omitempty"`
	OffsetBegin int64         `json:"offset_begin"`
	OffsetEnd   int64         `json:"offset_end"`
	Labels      []ResultLabel `json:"labels"`
	Cuts        []CutResult   `json:"cuts,omitempty"`
	Clips       []ClipResult  `json:"clips,omitempty"`
}

type EndResult struct {
	Code        int    `json:"code"`
	Message     string `json:"message"`
	OP          string `json:"op,omitempty"`
	OffsetBegin int64  `json:"offset_begin,omitempty"`
	OffsetEnd   int64  `json:"offset_end,omitempty"`
	Result      struct {
		Labels   []ResultLabel   `json:"labels"`
		Segments []SegmentResult `json:"segments"`
	} `json:"result"`
}

////////////////////////////////////////////////////////////////////////////////

type OPConfig struct {
	Host      string   `json:"host"`
	Timeout   int64    `json:"timeout"`
	Instances []string `json:"instances"`
	Params    OPParams `json:"params"`
}
