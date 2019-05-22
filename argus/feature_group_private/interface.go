package feature_group

import (
	"context"
	"encoding/json"

	"qiniu.com/argus/feature_group_private/proto"
)

const (
	DEFAULT_FEATURE_TAG = "default"
	DEFAULT_LIST_LIMIT  = 1000
)

type IGroups interface {
	New(context.Context, bool, proto.GroupName, proto.GroupConfig) error
	Get(context.Context, proto.GroupName) (IGroup, error)
	All(context.Context) ([]proto.GroupName, error)
}

type IGroup interface {
	Destroy(context.Context, bool) error
	Count(context.Context) (int, error)
	CountTags(context.Context) (int, error)
	Config(context.Context) proto.GroupConfig

	Add(context.Context, bool, ...proto.Feature) error
	Delete(context.Context, bool, ...proto.FeatureID) ([]proto.FeatureID, error)
	Update(context.Context, bool, ...proto.Feature) error
	Tags(ctx context.Context, marker string, limit int) ([]proto.GroupTagInfo, string, error)
	FilterByTag(ctx context.Context, tag proto.FeatureTag, marker string, limit int) ([]proto.Feature, string, error)

	Search(ctx context.Context, internal bool,
		threshold float32, limit int,
		features ...proto.FeatureValue,
	) (
		[][]FeatureSearchRawRespItem,
		error,
	)
	Compare(ctx context.Context,
		Target IGroup,
		threshold float32, limit int,
	) (
		[]BaseCompareResult,
		error,
	)
}

type IImageGroups interface {
	New(context.Context, proto.GroupName, proto.GroupConfig) error
	Get(context.Context, proto.GroupName) (IImageGroup, error)
	All(context.Context) ([]proto.GroupName, error)
}

type IImageGroup interface {
	AddImage(context.Context, ...proto.Image) error
	UpdateImage(context.Context, ...proto.Image) error
	SearchImage(ctx context.Context,
		threshold float32, limit int,
		images ...proto.ImageURI,
	) (
		[][]ImageSearchRespItem,
		error,
	)
	ListImage(ctx context.Context,
		tag proto.FeatureTag,
		marker string,
		limit int,
	) (
		images []ImageListRespItem,
		nextMarker string,
		err error,
	)
}

type IFaceGroups interface {
	New(context.Context, proto.GroupName, proto.GroupConfig) error
	Get(context.Context, proto.GroupName) (IFaceGroup, error)
	All(context.Context) ([]proto.GroupName, error)
	DetectAndFetchFeature(context.Context, bool, []proto.Data) (fvs [][]proto.FeatureValue, faceBoxes [][]proto.BoundingBox, err error)
}

type IFaceGroup interface {
	AddFace(context.Context, bool, ...proto.Image) ([]proto.FaceDetectBox, int, error)
	UpdateFace(context.Context, bool, ...proto.Image) error
	SearchFace(ctx context.Context,
		threshold float32, limit int,
		// TODO Face detect params
		// size_threshold [2]int
		fvs [][]proto.FeatureValue,
	) (
		[][]FaceSearchRespItem,
		error,
	)
	ListImage(ctx context.Context,
		tag proto.FeatureTag,
		marker string,
		limit int,
	) (
		images []ImageListRespItem,
		nextMarker string,
		err error,
	)
	AddFeature(ctx context.Context, features ...proto.Feature) (err error)
}

//------------------ RespItem ------------------//

type GroupsGetRespItem struct {
	Groups []proto.GroupName `json:"groups"`
}

type GroupGetRespItem struct {
	Config proto.GroupConfig `json:"config"`
	Tags   int               `json:"tags"`
	Count  int               `json:"count"`
}

type GroupGetTagsListResp struct {
	Tags   []proto.GroupTagInfo `json:"tags"`
	Marker string               `json:"marker"`
}

type FeatureSearchItem struct {
	ID    proto.FeatureID `json:"id"`
	Score float32         `json:"score"`
}

type FeatureSearchRawRespItem struct {
	Value proto.Feature `json:"value"`
	Score float32       `json:"score"`
}

type FeatureSearchRespItem struct {
	Value proto.FeatureJson `json:"value"`
	Score float32           `json:"score"`
}

type FeatureCompareItem struct {
	ID    proto.FeatureID     `json:"id"`
	Faces []FeatureSearchItem `json:"faces"`
}
type FeatureCompareRespItem struct {
	ID          proto.FeatureID   `json:"id"`
	Score       float32           `json:"score"`
	Tag         proto.FeatureTag  `json:"tag"`
	Desc        json.RawMessage   `json:"desc,omitempty"`
	BoundingBox proto.BoundingBox `json:"bounding_box,omitempty"`
}

type ImageSearchRespItem struct {
	ID    proto.FeatureID  `json:"id"`
	Score float32          `json:"score"`
	Tag   proto.FeatureTag `json:"tag"`
	Desc  json.RawMessage  `json:"desc,omitempty"`
}

type ImageSearchRespResults struct {
	Results []ImageSearchRespItem `json:"results"`
}

type ImageSearchResp struct {
	SearchResults []ImageSearchRespResults `json:"search_results"`
}

type ImageListRespItem struct {
	ID          proto.FeatureID   `json:"id"`
	Tag         proto.FeatureTag  `json:"tag"`
	Desc        json.RawMessage   `json:"desc,omitempty"`
	BoundingBox proto.BoundingBox `json:"bounding_box,omitempty"`
}

type ImageListResp struct {
	Images []ImageListRespItem `json:"images"`
	Marker string              `json:"marker"`
}

type FaceSearchRespFaceItem struct {
	ID          proto.FeatureID   `json:"id"`
	Score       float32           `json:"score"`
	Tag         proto.FeatureTag  `json:"tag"`
	Desc        json.RawMessage   `json:"desc,omitempty"`
	Group       proto.GroupName   `json:"group,omitempty"`
	BoundingBox proto.BoundingBox `json:"bounding_box,omitempty"`
}

type FaceSearchRespItem struct {
	BoundingBox proto.BoundingBox        `json:"bounding_box"`
	Faces       []FaceSearchRespFaceItem `json:"faces"`
}

type FaceSearchRespResults struct {
	Faces []FaceSearchRespItem `json:"faces"`
}

type FaceSearchResp struct {
	SearchResults []FaceSearchRespResults `json:"search_results"`
}

type BaseDeleteResp struct {
	Deleted []proto.FeatureID `json:"deleted"`
}

type BaseCompareResult struct {
	ID          proto.FeatureID          `json:"id"`
	Tag         proto.FeatureTag         `json:"tag"`
	Desc        json.RawMessage          `json:"desc,omitempty"`
	BoundingBox proto.BoundingBox        `json:"bounding_box,omitempty"`
	Faces       []FeatureCompareRespItem `json:"faces"`
}

type BaseCompareResp struct {
	CompareResults []BaseCompareResult `json:"compare_results"`
}
