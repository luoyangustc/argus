package imageg

import (
	"encoding/json"

	FG "qiniu.com/argus/feature_group"
)

//struct for http request & response
type ImageGroupAddReq struct {
	CmdArgs []string
	Data    []ImageGroupAddData `json:"data"`
}

type ImageGroupAddData struct {
	URI       string `json:"uri"`
	Attribute struct {
		ID    string          `json:"id"`
		Label string          `json:"label"`
		Desc  json.RawMessage `json:"desc,omitempty"`
	} `json:"attribute"`
}

type ImageGroupAddResp struct {
	Images []string `json:"images"`
	Errors []*struct {
		Code    int    `json:"code"`
		Message string `json:"message"`
	} `json:"errors"`
}

type ImageGroupSearchReq struct {
	CmdArgs []string
	Data    struct {
		URI string `json:"uri"`
	} `json:"data"`
	Params struct {
		Limit int `json:"limit"`
	} `json:"params"`
}

type ImageGroupSearchResp struct {
	Code    int                      `json:"code"`
	Message string                   `json:"message"`
	Result  []ImageGroupSearchResult `json:"result"`
}

type ImageGroupSearchResult struct {
	ID    string          `json:"id"`
	Label string          `json:"label"`
	Etag  string          `json:"etag"`
	URI   string          `json:"uri"`
	Score float32         `json:"score"`
	Desc  json.RawMessage `json:"desc,omitempty"`
}

type ImageGroupsSearchReq struct {
	Data struct {
		URI string `json:"uri"`
	} `json:"data"`
	Params ImageGroupsSearchReqParams `json:"params"`
}

type ImageGroupsSearchReqParams struct {
	Groups    []string `json:"groups"`
	Limit     int      `json:"limit"`
	Threshold float32  `json:"threshold"`
}

type ImageGroupsSearchResp struct {
	Code    int                       `json:"code"`
	Message string                    `json:"message"`
	Result  []ImageGroupsSearchResult `json:"result"`
}

type ImageGroupsSearchResult struct {
	ID    string          `json:"id"`
	Label string          `json:"label"`
	Etag  string          `json:"etag"`
	URI   string          `json:"uri"`
	Score float32         `json:"score"`
	Desc  json.RawMessage `json:"desc,omitempty"`
	Group string          `json:"group"`
}

//inner struct
type _IGSearchBlock struct {
	FeatureBlock  FG.FeatureBlock
	ImageFeature  *string
	FeatureLength int
	Threshold     float32
	_IGSearchGroupHub
}

type _IGSearchResultItem struct {
	FG.SearchResultItem
	_IGSearchGroupHub
}

type _IGSearchGroupHub struct {
	ImageGroup _ImageGroup
	Gid        string
	Hub        FG.Hub
	Hid        FG.HubID
}
