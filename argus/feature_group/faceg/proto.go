package faceg

import (
	"encoding/json"

	FG "qiniu.com/argus/feature_group"
	"qiniu.com/argus/utility"
)

//struct for http request & response
type FaceGroupAddReq struct {
	CmdArgs []string
	Data    []FaceGroupAddData `json:"data"`
}

type FaceGroupAddData struct {
	URI       string `json:"uri"`
	Attribute struct {
		ID   string          `json:"id"`
		Name string          `json:"name"`
		Mode string          `json:"mode"`
		Desc json.RawMessage `json:"desc,omitempty"`
	} `json:"attribute"`
}

type FaceGroupAddResp struct {
	Faces      []string `json:"faces"`
	Attributes []*struct {
		BoundingBox utility.FaceDetectBox `json:"bounding_box"`
	} `json:"attributes"`
	Errors []*struct {
		Code    int    `json:"code"`
		Message string `json:"message"`
	} `json:"errors"`
}

type FaceGroupSearchResp struct {
	Code    int                   `json:"code"`
	Message string                `json:"message"`
	Result  FaceGroupSearchResult `json:"result"`
}

type FaceGroupSearchResult struct {
	Review     bool                    `json:"review"`
	Detections []FaceGroupSearchDetail `json:"detections"`
}

type FaceGroupSearchDetail struct {
	BoundingBox utility.FaceDetectBox      `json:"boundingBox"`
	Value       FaceGroupSearchDetailValue `json:"value"`
}

type FaceGroupSearchDetailValue struct {
	BoundingBox utility.FaceDetectBox `json:"boundingBox,omitempty"`
	Name        string                `json:"name"`
	ID          string                `json:"id"`
	Score       float32               `json:"score"`
	Review      bool                  `json:"review"`
	Desc        json.RawMessage       `json:"desc,omitempty"`
}

type FaceGroupsSearchReq struct {
	Data struct {
		URI string `json:"uri"`
	} `json:"data"`
	Params FaceGroupsSearchReqParams `json:"params"`
}

type FaceGroupsSearchReqParams struct {
	Groups    []string `json:"groups"`
	Limit     int      `json:"limit"`
	Threshold float32  `json:"threshold"`
}

type FaceGroupsSearchResp struct {
	Code    int                    `json:"code"`
	Message string                 `json:"message"`
	Result  FaceGroupsSearchResult `json:"result"`
}

type FaceGroupsSearchResult struct {
	Review bool                     `json:"review"`
	Faces  []FaceGroupsSearchDetail `json:"faces"`
}

type FaceGroupsSearchDetail struct {
	BoundingBox utility.FaceDetectBox         `json:"bounding_box"`
	Faces       []FaceGroupsSearchDetailValue `json:"faces"`
}

type FaceGroupsSearchDetailValue struct {
	BoundingBox utility.FaceDetectBox `json:"bounding_box"`
	Name        string                `json:"name"`
	Group       string                `json:"group"`
	ID          string                `json:"id"`
	Score       float32               `json:"score"`
	Review      bool                  `json:"review"`
	Desc        json.RawMessage       `json:"desc,omitempty"`
}

//inner struct
type _FGSearchBlock struct {
	FeatureBlock  FG.FeatureBlock
	FaceFeature   *string
	FeatureLength int
	Threshold     float32
	_FGSearchGroupHub
}

type _FGSearchResultItem struct {
	FG.SearchResultItem
	_FGSearchGroupHub
}

type _FGSearchGroupHub struct {
	FaceGroup _FaceGroup
	Gid       string
	Hub       FG.Hub
	Hid       FG.HubID
}
