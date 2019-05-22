package vframe

const (
	MODE_INTERVAL = 0
	MODE_KEY      = 1
)

type VframeParams struct {
	Mode      *int    `json:"mode,omitempty"`
	Interval  float64 `json:"interval"`
	StartTime float64 `json:"ss,omitempty"`
	Duration  float64 `json:"t,omitempty"`
}

func (p VframeParams) GetMode() int {
	if p.Mode == nil {
		return MODE_INTERVAL
	}
	return *p.Mode
}

type LiveParams struct {
	Timeout   float64 `json:"timeout"`
	Downsteam string  `json:"downstream"`
}

type VframeRequest struct {
	Data struct {
		URI       string `json:"uri"`
		Attribute struct {
			ID string `json:"id"`
			// Meta json.RawMessage `json:"meta,omitempty"`
		} `json:"attribute"`
	} `json:"data"`
	Params VframeParams `json:"params,omitempty"`
	Live   *LiveParams  `json:"live,omitempty"`
}

// CutResponse ...
type CutResponse struct {
	Code    int    `json:"code"`
	Message string `json:"message"`
	Result  struct {
		Origin struct {
			URI string `json:"uri"`
			ID  string `json:"id"`
		} `json:"origin"`
		Cut struct {
			Offset int64  `json:"offset"`
			URI    string `json:"uri"`
		} `json:"cuts"`
	} `json:"result"`
}

type EndResponse struct {
	Code    int    `json:"code"`
	Message string `json:"message"`
	Result  struct {
		Origin struct {
			URI string `json:"uri"`
			ID  string `json:"id"`
		} `json:"origin"`
		CutCount int `json:"cut_count"`
	} `json:"result"`
}
