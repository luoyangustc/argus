package proto

import (
	"context"
	"time"

	"qiniu.com/auth/authstub.v1"
)

// 最终暴露给用户API
type UserApi interface {
	// 添加图片
	PostImage(ctx context.Context, req *PostImageReq, env *authstub.Env) (resp *PostImageResp, err error)
	// 提交搜索任务
	PostSearchJob(ctx context.Context, req *PostSearchJobReq, env *authstub.Env) (resp *PostSearchJobResp, err error)
	// 查询搜索任务
	GetSearchJob(ctx context.Context, req *GetSearchJobReq, env *authstub.Env) (resp *GetSearchJobResp, err error)
	// 创建hub
	PostHub(ctx context.Context, req *PostHubReq, env *authstub.Env) (err error)
	GetHubs(ctx context.Context, req *GetHubsReq, env *authstub.Env) (resp *GetHubsResp, err error)
	GetHub(ctx context.Context, req *GetHubReq, env *authstub.Env) (resp *GetHubResp, err error)
}

// 图片特征提取算法模块
type ImageFeatureApi interface {
	PostEvalFeature(ctx context.Context, req PostEvalFeatureReq) (resp *PostEvalFeatureResp, err error)
}

type Feature []byte //16K

type PostImageReq struct {
	Images []ImageKey `json:"images"`
	// 一个Hub对应一个uid下面的bucket
	Hub string `json:"hub"`
	Op  string `json:"op"`
}

type ImageKey struct {
	Key string `json:"key"`
}

type ImageKeyUrl struct {
	Key string `json:"key"`
	Url string `json:"url"`
}

type Image struct {
	Key    string `json:"key"`
	Bucket string `json:"bucket"`
	Uid    uint32 `json:"uid"`
	Url    string `json:"url"`
}

type PostImageResp struct {
	SuccessCnt  int `json:"success_cnt"`
	NotFoundCnt int `json:"not_found_cnt"`
	ExistsCnt   int `json:"exists_cnt"`
}

type SearchKind int

const (
	TopNSearch SearchKind = iota
	ThresholdSearch
)

type PostSearchJobReq struct {
	Images      []ImageKeyUrl `json:"images"`
	Hub         string        `json:"hub"`
	TopN        int           `json:"topN"`
	Threshold   float32       `json:"threshold"`
	Kind        SearchKind    `json:"kind"`
	CallBackURL string        `json:"callback_url"`
}

type PostSearchJobReqJob struct {
	Images      []Image    `json:"images"`
	Hub         string     `json:"hub"`
	Version     int        `json:"version"`
	TopN        int        `json:"topN"`
	Threshold   float32    `json:"threshold"`
	Kind        SearchKind `json:"kind"`
	CallBackURL string     `json:"callback_url"`
}

type PostSearchJobResp struct {
	JobID string `json:"job_id"`
}

type GetSearchJobReq PostSearchJobResp

type GetSearchJobResp struct {
	Status string                 `json:"status"`
	Images []SearchImageRespImage `json:"images"`
}

type ErrorMsg struct {
	Msg string `json:"msg"`
}
type SearchImageRespImage struct {
	OriginImage Image `json:"origin_image"`
	Result      struct {
		Keys []string `json:"keys"`
		Err  ErrorMsg `json:"err"`
	} `json:"result"`
}

type PostHubReq struct {
	Name   string `json:"name"`
	Bucket string `json:"bucket"`
	Prefix string `json:"prefix"`
}

type PostEvalFeatureReq struct {
	Image Image `json:"image"`
}

type PostEvalFeatureResp struct {
	Feature Feature `json:"feature"` // 16k
	Md5     string  `json:"md5"`
}

type FeatureItem struct {
	Feature Feature `json:"feature"`
	Index   int     `json:"index"`
	Offset  int     `json:"offset"`
}

type DistanceItem struct {
	Distance float32 `json:"distance"`
	Index    int     `json:"index"`
	Offset   int     `json:"offset"`
	Key      string  `json:"key"` // filemeta
}

type GetHubsReq struct {
}

type GetHubsResp struct {
	Hubs []Hub `json:"hubs"`
}

type GetHubReq struct {
	Hub string `json:"hub"`
}

type GetHubResp struct {
	HubName string         `json:"hub_name"`
	Bucket  string         `json:"bucket"`
	Prefix  string         `json:"prefix"`
	Stat    GetHubRespStat `json:"stat"`
}
type GetHubRespStat struct {
	ImageNum int `json:"image_num"`
}

type Hub struct {
	HubName string `json:"hub_name"`
	Bucket  string `json:"bucket"`
	Prefix  string `json:"prefix"`
}

type InternalApi interface {
	GetHubInfo(ctx context.Context, req *GetHubInfoReq, env *authstub.Env) (resp *GetHubInfoResp, err error)
	GetFilemetaInfo(context context.Context, req *GetFileMetaInfoReq, env *authstub.Env) (*GetFileMetaInfoResp, error)
}

type GetHubInfoReq struct {
	HubName string `json:"name"`
	Version int    `json:"version"`
}

type GetHubInfoResp struct {
	Uid              uint32 `json:"uid"`
	Bucket           string `json:"bucket"`
	Prefix           string `json:"prefix"`
	FeatureFileIndex int    `json:"index"`
}

type GetFileMetaInfoReq struct {
	HubName           string `json:"name"`
	FeatureFileIndex  int    `json:"index"`
	FeatureFileOffset int    `json:"offset"`
}

type GetFileMetaInfoResp struct {
	Key        string    `json:"key"`
	UpdateTime time.Time `json:"update_time"`
	Status     string    `json:"status"`
	Md5        string    `json:"md5,omitempty"`
}
