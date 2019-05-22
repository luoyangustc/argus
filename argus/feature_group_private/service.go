package feature_group

import (
	"context"

	"github.com/qiniu/http/restrpc.v1"

	"qiniu.com/argus/feature_group_private/proto"
)

// 修改此文件请和 docs/Argus/feature_group_private.md 同步

type IGroupService interface {
	PostGroups_(context.Context,
		*struct {
			CmdArgs         []string
			Config          proto.GroupConfig `json:"config"`
			ClusterInternal bool              `json:"cluster_internal"`
		},
		*restrpc.Env,
	) error
	GetGroups_(context.Context, *struct{ CmdArgs []string }, *restrpc.Env) (
		GroupGetRespItem,
		error)
	GetGroups(context.Context, *struct{ CmdArgs []string }, *restrpc.Env) (
		GroupsGetRespItem,
		error)
	PostGroups_Remove(context.Context,
		*struct {
			CmdArgs         []string
			ClusterInternal bool `json:"cluster_internal"`
		}, *restrpc.Env) error
	PostGroups_Delete(context.Context,
		*struct {
			CmdArgs         []string
			IDs             []proto.FeatureID `json:"ids"`
			ClusterInternal bool              `json:"cluster_internal"`
		},
		*restrpc.Env,
	) (BaseDeleteResp, error)
	GetGroups_Tags(context.Context,
		*struct {
			CmdArgs []string
			Marker  string `json:"marker"`
			Limit   int    `json:"limit"`
		},
		*restrpc.Env,
	) (GroupGetTagsListResp, error)
	PostGroups_Compare(context.Context,
		*struct {
			CmdArgs     []string
			TargetGroup string  `json:"target_group"`
			Threshold   float32 `json:"threshold"`
			Limit       int     `json:"limit"`
		},
		*restrpc.Env,
	) (BaseCompareResp, error)
}

type IFeatueGroupService interface {
	PostGroups_FeatureAdd(context.Context,
		*struct {
			CmdArgs         []string
			Features        []proto.FeatureJson `json:"features"`
			ClusterInternal bool                `json:"cluster_internal"`
		},
		*restrpc.Env,
	) error
	PostGroups_FeatureUpdate(context.Context,
		*struct {
			CmdArgs         []string
			Features        []proto.FeatureJson `json:"features"`
			ClusterInternal bool                `json:"cluster_internal"`
		},
		*restrpc.Env,
	) error
	PostGroups_FeatureSearch(context.Context,
		*struct {
			CmdArgs         []string
			Features        []proto.FeatureValueJson `json:"features"`
			Threshold       float32                  `json:"threshold"`
			Limit           int                      `json:"limit"`
			ClusterInternal bool                     `json:"cluster_internal"`
		},
		*restrpc.Env,
	) (
		[][]FeatureSearchRespItem,
		error)
}

type IImageGroupService interface {
	PostGroups_Add(context.Context,
		*struct {
			CmdArgs []string
			Image   proto.Image `json:"image"`
		},
		*restrpc.Env,
	) (*struct {
		ID proto.FeatureID `json:"id"`
	}, error)
	PostGroups_Update(context.Context,
		*struct {
			CmdArgs []string
			Image   proto.Image `json:"image"`
		},
		*restrpc.Env,
	) error
	PostGroups_Search(context.Context,
		*struct {
			CmdArgs   []string
			Images    []proto.ImageURI `json:"images"`
			Threshold float32          `json:"threshold"`
			Limit     int              `json:"limit"`
		},
		*restrpc.Env,
	) (
		ImageSearchResp,
		error)
	GetGroups_Images(context.Context,
		*struct {
			CmdArgs []string
			Tag     proto.FeatureTag `json:"tag"`
			Marker  string           `json:"marker"`
			Limit   int              `json:"limit"`
		},
		*restrpc.Env,
	) (ImageListResp, error)
}

type IFaceGroupService interface {
	PostGroups_Add(context.Context,
		*struct {
			CmdArgs []string
			Image   proto.Image `json:"image"`
			Params  struct {
				RejectBadFace bool `json:"reject_bad_face"`
			} `json:"params,omitempty"`
		},
		*restrpc.Env,
	) (*struct {
		ID          proto.FeatureID   `json:"id"`
		BoundingBox proto.BoundingBox `json:"bounding_box"`
		FaceQuality proto.FaceQuality `json:"face_quality,omitempty"`
		FaceNumber  int               `json:"face_number,omitempty"`
	}, error)
	PostGroups_Update(context.Context,
		*struct {
			CmdArgs []string
			Image   proto.Image `json:"image"`
			Params  struct {
				RejectBadFace bool `json:"reject_bad_face"`
			} `json:"params,omitempty"`
		},
		*restrpc.Env,
	) error
	PostGroups_Search(context.Context,
		*struct {
			CmdArgs    []string
			Images     []proto.ImageURI `json:"images"`
			Threshold  float32          `json:"threshold"`
			Limit      int              `json:"limit"`
			UseQuality bool             `json:"use_quality"`
		},
		*restrpc.Env,
	) (
		FaceSearchResp,
		error)
	PostGroupsMultiSearch(context.Context,
		*struct {
			Images       []proto.ImageURI  `json:"images"`
			Groups       []proto.GroupName `json:"groups"`
			ClusterGroup proto.GroupName   `json:"cluster_group"`
			Threshold    float32           `json:"threshold"`
			Limit        int               `json:"limit"`
			UseQuality   bool              `json:"use_quality"`
		},
		*restrpc.Env,
	) (
		FaceSearchResp,
		error)
	GetGroups_Images(context.Context,
		*struct {
			CmdArgs []string
			Tag     proto.FeatureTag `json:"tag"`
			Marker  string           `json:"marker"`
			Limit   int              `json:"limit"`
		},
		*restrpc.Env,
	) (ImageListResp, error)
	PostGroups_Cluster(ctx context.Context,
		args *struct {
			CmdArgs    []string
			Images     []proto.ImageURI `json:"images"`
			Threshold  float32          `json:"threshold"`
			Limit      int              `json:"limit"`
			UseQuality bool             `json:"use_quality"`
		},
		env *restrpc.Env,
	) (FaceSearchResp, error)

	PostGroups_PtsSearch(ctx context.Context,
		args *struct {
			CmdArgs []string
			Data    []proto.Data `json:"data"`
			Params  struct {
				Threshold  float32 `json:"threshold"`
				Limit      int     `json:"limit"`
				UseQuality bool    `json:"use_quality,omitempty"`
			} `json:"params,omitempty"`
		}, env *restrpc.Env,
	) (FaceSearchResp, error)

	PostGroupsMultiPtsSearch(ctx context.Context,
		args *struct {
			Data   []proto.Data `json:"data"`
			Params struct {
				Threshold  float32 `json:"threshold"`
				Limit      int     `json:"limit"`
				UseQuality bool    `json:"use_quality,omitempty"`
			} `json:"params,omitempty"`
			Groups       []proto.GroupName `json:"groups"`
			ClusterGroup proto.GroupName   `json:"cluster_group"`
		},
		env *restrpc.Env,
	) (FaceSearchResp, error)
}
