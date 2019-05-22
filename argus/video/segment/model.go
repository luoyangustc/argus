package segment

type SegmentParams struct {
	Mode     int     `json:"mode"`
	Duration float64 `json:"duration"`
}

type SegmentRequest struct {
	Data struct {
		URI       string `json:"uri"`
		Attribute struct {
			ID string `json:"id"`
		} `json:"attribute"`
	} `json:"data"`
	Params SegmentParams `json:"params,omitempty"`
}

type ClipResponse struct {
	Code    int    `json:"code"`
	Message string `json:"message"`
	Result  struct {
		Origin struct {
			URI string `json:"uri"`
			ID  string `json:"id"`
		} `json:"origin"`
		Clip struct {
			OffsetBegin int64  `json:"offset_begin"`
			OffsetEnd   int64  `json:"offset_end"`
			URI         string `json:"uri"`
		} `json:"clips"`
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
		ClipCount int `json:"clip_count"`
	} `json:"result"`
}
