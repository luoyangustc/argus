package service

import (
	"context"
	"fmt"
	"hash/crc32"
	"net/http"
	"regexp"

	"github.com/qiniu/http/httputil.v1"

	"github.com/qiniu/http/restrpc.v1"
	xlog "github.com/qiniu/xlog.v1"

	"qiniu.com/argus/feature_group_private"
	"qiniu.com/argus/feature_group_private/manager"
	"qiniu.com/argus/feature_group_private/proto"
)

var _ feature_group.IGroupService = new(BaseService)
var _ feature_group.IFeatueGroupService = new(BaseService)

type BaseService struct {
	Config BaseGroupsConfig
	groups feature_group.IGroups
}

func NewBaseService(ctx context.Context, groups feature_group.IGroups, config BaseGroupsConfig) (interface {
	feature_group.IGroupService
	feature_group.IFeatueGroupService
}, error) {
	s := &BaseService{
		groups: groups,
		Config: config,
	}
	return s, nil
}

func (s *BaseService) initContext(ctx context.Context, env *restrpc.Env) (*xlog.Logger, context.Context) {
	xl, ok := xlog.FromContext(ctx)
	if !ok {
		xl = xlog.New(env.W, env.Req)
		ctx = xlog.NewContext(ctx, xl)
	}
	return xl, ctx
}

func (s *BaseService) PostGroups_(ctx context.Context,
	args *struct {
		CmdArgs         []string
		Config          proto.GroupConfig `json:"config"`
		ClusterInternal bool              `json:"cluster_internal"`
	},
	env *restrpc.Env,
) error {
	var err error
	xl, ctx := s.initContext(ctx, env)

	groupName := args.CmdArgs[0]
	if len(groupName) == 0 {
		return httputil.NewError(http.StatusBadRequest, "invalid arguments")
	}
	if !regexp.MustCompile(`^[a-z][a-z0-9_-]{2,31}$`).MatchString(groupName) {
		return httputil.NewError(http.StatusBadRequest, "invalid group name")
	}
	if !args.ClusterInternal {
		group, err := s.groups.Get(ctx, proto.GroupName(groupName))
		if err != nil && err != manager.ErrGroupNotExist {
			xl.Error("failed to check group exist", err)
			return httputil.NewError(http.StatusInternalServerError, "failed to check group exist")
		}
		if group != nil {
			return httputil.NewError(http.StatusConflict, "group already exist")
		}
	}

	if args.Config.Capacity <= 0 {
		return httputil.NewError(http.StatusBadRequest, "invalid group capacity")
	}

	if args.Config.Dimension == 0 {
		args.Config.Dimension = s.Config.Sets.Dimension

	}
	if args.Config.Precision == 0 {
		args.Config.Precision = s.Config.Sets.Precision
	}

	err = s.groups.New(ctx, args.ClusterInternal, proto.GroupName(groupName), args.Config)
	if err != nil {
		xl.Errorf("PostGroups_ groups.New %s failed: %v", groupName, err)
		return httputil.NewError(http.StatusInternalServerError, "failed to create group")
	}
	xl.Infof("Create group %s with config %v", groupName, args.Config)
	return nil
}

func (s *BaseService) GetGroups(ctx context.Context, args *struct{ CmdArgs []string }, env *restrpc.Env) (
	resp feature_group.GroupsGetRespItem, err error) {
	xl, ctx := s.initContext(ctx, env)
	if resp.Groups, err = s.groups.All(ctx); err != nil {
		xl.Errorf("groups.All failed, err: %s", err)
		return resp, httputil.NewError(http.StatusInternalServerError, "fail to search group")
	}
	return
}

func (s *BaseService) GetGroups_(ctx context.Context, args *struct{ CmdArgs []string }, env *restrpc.Env) (
	resp feature_group.GroupGetRespItem, err error) {
	xl, ctx := s.initContext(ctx, env)

	groupName := args.CmdArgs[0]
	if len(groupName) == 0 {
		return resp, httputil.NewError(http.StatusBadRequest, "invalid arguments")
	}
	group, err := s.groups.Get(ctx, proto.GroupName(groupName))
	if err != nil {
		xl.Error("GetGroups_ groups.Get failed:", err)
		return resp, httputil.NewError(http.StatusBadRequest, "group not exist")
	}
	resp.Config = group.Config(ctx)
	count, err := group.Count(ctx)
	if err != nil {
		xl.Errorf("GetGroups_ %s count failed, error: %s", groupName, err.Error())
		return resp, httputil.NewError(http.StatusInternalServerError, "failed to count group")
	}
	resp.Count = count
	tagCount, err := group.CountTags(ctx)
	if err != nil {
		xl.Errorf("GetGroups_ count %s tags failed, error: %s", groupName, err.Error())
		return resp, httputil.NewError(http.StatusInternalServerError, "failed to tag group")
	}
	resp.Tags = tagCount
	xl.Debugf("Get grop %s", groupName)
	return
}

func (s *BaseService) PostGroups_Remove(ctx context.Context,
	args *struct {
		CmdArgs         []string
		ClusterInternal bool `json:"cluster_internal"`
	},
	env *restrpc.Env) error {
	xl, ctx := s.initContext(ctx, env)

	groupName := args.CmdArgs[0]
	if len(groupName) == 0 {
		return httputil.NewError(http.StatusBadRequest, "invalid arguments")
	}

	group, err := s.groups.Get(ctx, proto.GroupName(groupName))
	if err != nil {
		xl.Error("PostGroups_Remove groups.Get failed:", err)
		return httputil.NewError(http.StatusBadRequest, "group not exist")
	}
	err = group.Destroy(ctx, args.ClusterInternal)
	if err != nil {
		xl.Error("PostGroups_Remove group.Destroy failed:", err)
		return httputil.NewError(http.StatusInternalServerError, "failed to remove group")
	}
	xl.Debugf("Remove group %s", groupName)
	return nil
}

func (s *BaseService) PostGroups_Delete(ctx context.Context,
	args *struct {
		CmdArgs         []string
		IDs             []proto.FeatureID `json:"ids"`
		ClusterInternal bool              `json:"cluster_internal"`
	},
	env *restrpc.Env,
) (feature_group.BaseDeleteResp, error) {
	xl, ctx := s.initContext(ctx, env)
	resp := feature_group.BaseDeleteResp{}

	groupName := args.CmdArgs[0]
	if len(groupName) == 0 {
		return resp, httputil.NewError(http.StatusBadRequest, "invalid arguments")
	}

	group, err := s.groups.Get(ctx, proto.GroupName(groupName))
	if err != nil {
		xl.Error("PostGroups_Delete groups.Get failed:", err)
		return resp, httputil.NewError(http.StatusBadRequest, "group not exist")
	}
	resp.Deleted, err = group.Delete(ctx, args.ClusterInternal, args.IDs...)
	if err != nil {
		xl.Error("PostGroups_Delete group.Delete failed:", err)
		return resp, httputil.NewError(http.StatusInternalServerError, "failed to delete feature")
	}
	xl.Debugf("Group %s delete features: %v", groupName, args.IDs)
	return resp, nil
}

func (s *BaseService) PostGroups_FeatureAdd(ctx context.Context,
	args *struct {
		CmdArgs         []string
		Features        []proto.FeatureJson `json:"features"`
		ClusterInternal bool                `json:"cluster_internal"`
	},
	env *restrpc.Env,
) error {
	xl, ctx := s.initContext(ctx, env)

	groupName := args.CmdArgs[0]
	if len(groupName) == 0 {
		return httputil.NewError(http.StatusBadRequest, "invalid arguments")
	}

	group, err := s.groups.Get(ctx, proto.GroupName(groupName))
	if err != nil {
		xl.Error("PostGroups_FeatureAdd groups.Get failed:", err)
		return httputil.NewError(http.StatusBadRequest, "group not exist")
	}
	config := group.Config(ctx)
	featureLen := config.Dimension * config.Precision

	features := []proto.Feature{}
	for _, fj := range args.Features {
		fjv := fj.ToFeature()
		if len(fjv.Value) != featureLen {
			xl.Error("PostGroups_FeatureAdd feature format invalid")
			return httputil.NewError(http.StatusBadRequest, "invalid feature format")
		}
		fjv.HashKey = proto.FeatureHashKey(crc32.ChecksumIEEE([]byte(fjv.ID)))
		features = append(features, fjv)
	}
	err = group.Add(ctx, args.ClusterInternal, features...)
	if err != nil {
		xl.Error("PostGroups_FeatureAdd group.Add", err)
		return httputil.NewError(http.StatusInternalServerError, "failed to add feature")
	}
	xl.Debugf("Group %s add %d features", groupName, len(args.Features))
	return nil
}

func (s *BaseService) PostGroups_FeatureUpdate(ctx context.Context,
	args *struct {
		CmdArgs         []string
		Features        []proto.FeatureJson `json:"features"`
		ClusterInternal bool                `json:"cluster_internal"`
	},
	env *restrpc.Env,
) error {
	xl, ctx := s.initContext(ctx, env)

	groupName := args.CmdArgs[0]
	if len(groupName) == 0 {
		return httputil.NewError(http.StatusBadRequest, "invalid arguments")
	}

	group, err := s.groups.Get(ctx, proto.GroupName(groupName))
	if err != nil {
		xl.Error("PostGroups_FeatureUpdate groups.Get failed:", err)
		return httputil.NewError(http.StatusBadRequest, "group not exist")
	}

	features := []proto.Feature{}
	for _, fj := range args.Features {
		features = append(features, fj.ToFeature())
	}
	err = group.Update(ctx, args.ClusterInternal, features...)
	if err != nil {
		xl.Error("PostGroups_FeatureUpdate group.Update", err)
		return httputil.NewError(http.StatusInternalServerError, "failed to update feature")
	}
	xl.Debugf("Group %s update features: %v", groupName, args.Features)
	return nil
}

func (s *BaseService) PostGroups_FeatureSearch(ctx context.Context,
	args *struct {
		CmdArgs         []string
		Features        []proto.FeatureValueJson `json:"features"`
		Threshold       float32                  `json:"threshold"`
		Limit           int                      `json:"limit"`
		ClusterInternal bool                     `json:"cluster_internal"`
	},
	env *restrpc.Env,
) (
	[][]feature_group.FeatureSearchRespItem,
	error) {
	xl, ctx := s.initContext(ctx, env)
	if args.Limit == 0 {
		args.Limit = defaultSearchLimit
	}
	resp := make([][]feature_group.FeatureSearchRespItem, 0)

	groupName := args.CmdArgs[0]
	if len(groupName) == 0 {
		return resp, httputil.NewError(http.StatusBadRequest, "invalid arguments")
	}

	group, err := s.groups.Get(ctx, proto.GroupName(groupName))
	if err != nil {
		xl.Error("PostGroups_FeatureSearch groups.Get failed:", err)
		return resp, httputil.NewError(http.StatusBadRequest, "group not exist")
	}
	featureValues := []proto.FeatureValue{}
	for _, fvj := range args.Features {
		featureValues = append(featureValues, fvj.ToFeatureValue())
	}
	rets, err := group.Search(ctx, args.ClusterInternal, args.Threshold, args.Limit, featureValues...)
	if err != nil {
		xl.Error("PostGroups_FeatureSearch group.Search", err)
		return resp, httputil.NewError(http.StatusInternalServerError, "failed to search feature")
	}

	for r, row := range rets {
		resp = append(resp, make([]feature_group.FeatureSearchRespItem, 0))
		for _, item := range row {
			resp[r] = append(resp[r], feature_group.FeatureSearchRespItem{
				Value: item.Value.ToFeatureJson(),
				Score: item.Score,
			})
		}
	}

	return resp, nil
}

func (s *BaseService) GetGroups_Tags(ctx context.Context,
	args *struct {
		CmdArgs []string
		Marker  string `json:"marker"`
		Limit   int    `json:"limit"`
	},
	env *restrpc.Env,
) (feature_group.GroupGetTagsListResp, error) {
	xl, ctx := s.initContext(ctx, env)
	resp := feature_group.GroupGetTagsListResp{}

	groupName := args.CmdArgs[0]
	if len(groupName) == 0 {
		return resp, httputil.NewError(http.StatusBadRequest, "invalid arguments")
	}
	if 0 >= args.Limit {
		args.Limit = feature_group.DEFAULT_LIST_LIMIT
	}

	group, err := s.groups.Get(ctx, proto.GroupName(groupName))
	if err != nil {
		xl.Error("GetGroups_Tags groups.Get failed: ", err)
		return resp, httputil.NewError(http.StatusBadRequest, "group not exist")
	}

	resp.Tags, resp.Marker, err = group.Tags(ctx, args.Marker, args.Limit)
	if err != nil {
		xl.Error("GetGroups_Tags group.Tags failed: ", err)
		return resp, httputil.NewError(http.StatusInternalServerError, "failed to list image")
	}

	xl.Debugf("Group %s list tags, marker %s, limit %d", groupName, args.Marker, args.Limit)
	return resp, nil
}

func (s *BaseService) PostGroups_Compare(ctx context.Context,
	args *struct {
		CmdArgs     []string
		TargetGroup string  `json:"target_group"`
		Threshold   float32 `json:"threshold"`
		Limit       int     `json:"limit"`
	},
	env *restrpc.Env,
) (feature_group.BaseCompareResp, error) {
	xl, ctx := s.initContext(ctx, env)
	var (
		resp      feature_group.BaseCompareResp
		groupName = args.CmdArgs[0]
		err       error
	)

	if len(groupName) == 0 {
		return resp, httputil.NewError(http.StatusBadRequest, "invalid face group name")
	}

	if len(args.TargetGroup) == 0 {
		return resp, httputil.NewError(http.StatusBadRequest, "invalid target face group name")
	}

	if args.Threshold < 0 || args.Threshold > 1 {
		return resp, httputil.NewError(http.StatusBadRequest, "invalid threshold")
	}
	if 0 >= args.Limit {
		args.Limit = feature_group.DEFAULT_LIST_LIMIT
	}

	group, err := s.groups.Get(ctx, proto.GroupName(groupName))
	if err != nil {
		xl.Error("PostGroups_Compare groups.Get failed: ", err)
		return resp, httputil.NewError(http.StatusBadRequest, "face group not exist")
	}

	targetGroup, err := s.groups.Get(ctx, proto.GroupName(args.TargetGroup))
	if err != nil {
		xl.Error("PostGroups_Compare target groups.Get failed: ", err)
		return resp, httputil.NewError(http.StatusBadRequest, "target face group not exist")
	}

	resp.CompareResults, err = group.Compare(ctx, targetGroup, args.Threshold, args.Limit)
	if err != nil {
		xl.Errorf("PostGroups_Compare group %s, target %s, compare error: %s", groupName, args.TargetGroup, err.Error())
		return resp, httputil.NewError(http.StatusInternalServerError, fmt.Sprintf("compare group failed, %s", err.Error()))
	}
	return resp, nil
}
