package service

import (
	"context"
	"net/http"

	"github.com/qiniu/http/httputil.v1"
	"github.com/qiniu/http/restrpc.v1"
	xlog "github.com/qiniu/xlog.v1"

	"github.com/qiniu/uuid"
	"qiniu.com/argus/feature_group_private"
	"qiniu.com/argus/feature_group_private/proto"
)

var _ feature_group.IImageGroupService = new(ImageService)

type ImageService struct {
	Config ImageGroupsConfig
	groups feature_group.IImageGroups
}

func NewImageService(ctx context.Context, baseGroups *BaseGroups, config ImageGroupsConfig) (feature_group.IImageGroupService, error) {
	xl := xlog.FromContextSafe(ctx)
	groups, err := NewImageGroups(ctx, baseGroups, config)
	if err != nil {
		xl.Error("NewImageGroups failed", err)
		return nil, err
	}
	s := &ImageService{
		groups: groups,
		Config: config,
	}
	return s, nil
}

func (s *ImageService) initContext(ctx context.Context, env *restrpc.Env) (*xlog.Logger, context.Context) {
	xl, ok := xlog.FromContext(ctx)
	if !ok {
		xl = xlog.New(env.W, env.Req)
		ctx = xlog.NewContext(ctx, xl)
	}
	return xl, ctx
}

func (s *ImageService) PostGroups_Add(ctx context.Context,
	args *struct {
		CmdArgs []string
		Image   proto.Image `json:"image"`
	},
	env *restrpc.Env,
) (*struct {
	ID proto.FeatureID `json:"id"`
}, error) {
	xl, ctx := s.initContext(ctx, env)

	groupName := args.CmdArgs[0]
	if len(groupName) == 0 {
		return nil, httputil.NewError(http.StatusBadRequest, "invalid arguments")
	}
	if args.Image.ID == "" {
		s, err := uuid.Gen(16)
		if err != nil {
			return nil, httputil.NewError(http.StatusInternalServerError, "failed to gen uuid")
		}
		args.Image.ID = proto.FeatureID(s)
	}

	if args.Image.Tag == "" {
		args.Image.Tag = feature_group.DEFAULT_FEATURE_TAG
	}

	group, err := s.groups.Get(ctx, proto.GroupName(groupName))
	if err != nil {
		xl.Error("PostGroups_Add groups.Get failed: ", err)
		return nil, httputil.NewError(http.StatusBadRequest, "group not exist")
	}
	err = group.AddImage(ctx, args.Image)
	if err != nil {
		xl.Error("PostGroups_Add group.AddImage failed: ", err)
		return nil, httputil.NewError(http.StatusInternalServerError, "failed to add image")
	}

	xl.Debugf("Group %v add %v image", groupName, args.Image.ID)
	return &struct {
		ID proto.FeatureID `json:"id"`
	}{ID: args.Image.ID}, nil
}

func (s *ImageService) PostGroups_Update(ctx context.Context,
	args *struct {
		CmdArgs []string
		Image   proto.Image `json:"image"`
	},
	env *restrpc.Env,
) error {
	xl, ctx := s.initContext(ctx, env)

	groupName := args.CmdArgs[0]
	if len(groupName) == 0 {
		return httputil.NewError(http.StatusBadRequest, "invalid arguments")
	}

	if args.Image.Tag == "" {
		args.Image.Tag = feature_group.DEFAULT_FEATURE_TAG
	}

	group, err := s.groups.Get(ctx, proto.GroupName(groupName))
	if err != nil {
		xl.Error("PostGroups_Update groups.Get failed: ", err)
		return httputil.NewError(http.StatusBadRequest, "group not exist")
	}
	err = group.UpdateImage(ctx, args.Image)
	if err != nil {
		xl.Error("PostGroups_Update group.UpdateImage failed: ", err)
		return httputil.NewError(http.StatusInternalServerError, "failed to update image")
	}

	xl.Debugf("Group %v update %v image", groupName, args.Image.ID)
	return nil
}

func (s *ImageService) PostGroups_Search(ctx context.Context,
	args *struct {
		CmdArgs   []string
		Images    []proto.ImageURI `json:"images"`
		Threshold float32          `json:"threshold"`
		Limit     int              `json:"limit"`
	},
	env *restrpc.Env,
) (feature_group.ImageSearchResp, error) {
	xl, ctx := s.initContext(ctx, env)
	resp := feature_group.ImageSearchResp{}

	groupName := args.CmdArgs[0]
	if len(groupName) == 0 {
		return resp, httputil.NewError(http.StatusBadRequest, "invalid arguments")
	}
	if args.Threshold < 0 || args.Threshold > 1 {
		return resp, httputil.NewError(http.StatusBadRequest, "invalid threshold")
	}
	if 0 >= args.Limit {
		args.Limit = feature_group.DEFAULT_LIST_LIMIT
	}

	group, err := s.groups.Get(ctx, proto.GroupName(groupName))
	if err != nil {
		xl.Error("PostGroups_Search groups.Get failed: ", err)
		return resp, httputil.NewError(http.StatusBadRequest, "group not exist")
	}
	results, err := group.SearchImage(ctx, args.Threshold, args.Limit, args.Images...)
	if err != nil {
		xl.Error("PostGroups_Search group.SearchImage failed: ", err)
		return resp, httputil.NewError(http.StatusInternalServerError, "failed to search image")
	}

	for _, row := range results {
		imageSearchRespResults := feature_group.ImageSearchRespResults{}
		imageSearchRespResults.Results = append(imageSearchRespResults.Results, row...)
		if len(row) == 0 {
			imageSearchRespResults.Results = make([]feature_group.ImageSearchRespItem, 0)
		}
		resp.SearchResults = append(resp.SearchResults, imageSearchRespResults)
	}
	if len(results) == 0 {
		imageSearchRespResults := feature_group.ImageSearchRespResults{}
		imageSearchRespResults.Results = make([]feature_group.ImageSearchRespItem, 0)
		resp.SearchResults = append(resp.SearchResults, imageSearchRespResults)
	}
	xl.Debugf("Group %s search images", groupName)
	return resp, nil
}

func (s *ImageService) GetGroups_Images(ctx context.Context,
	args *struct {
		CmdArgs []string
		Tag     proto.FeatureTag `json:"tag"`
		Marker  string           `json:"marker"`
		Limit   int              `json:"limit"`
	},
	env *restrpc.Env,
) (feature_group.ImageListResp, error) {
	xl, ctx := s.initContext(ctx, env)
	resp := feature_group.ImageListResp{}

	groupName := args.CmdArgs[0]
	if len(groupName) == 0 {
		return resp, httputil.NewError(http.StatusBadRequest, "invalid arguments")
	}
	if 0 >= args.Limit {
		args.Limit = feature_group.DEFAULT_LIST_LIMIT
	}

	group, err := s.groups.Get(ctx, proto.GroupName(groupName))
	if err != nil {
		xl.Error("GetGroups_Images groups.Get failed: ", err)
		return resp, httputil.NewError(http.StatusBadRequest, "group not exist")
	}

	resp.Images, resp.Marker, err = group.ListImage(ctx, args.Tag, args.Marker, args.Limit)
	if err != nil {
		xl.Error("GetGroups_Images group.ListImages failed: ", err)
		return resp, httputil.NewError(http.StatusInternalServerError, "failed to list image")
	}

	xl.Debugf("Group %s list images, tag %s marker %s, limit %d", groupName, args.Tag, args.Marker, args.Limit)
	return resp, nil
}
