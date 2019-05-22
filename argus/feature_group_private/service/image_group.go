package service

import (
	"context"
	"encoding/base64"
	"hash/crc32"
	"time"

	"github.com/pkg/errors"
	xlog "github.com/qiniu/xlog.v1"

	"qiniu.com/argus/feature_group_private"
	"qiniu.com/argus/feature_group_private/feature"
	"qiniu.com/argus/feature_group_private/proto"
)

type ImageGroupsConfig struct {
	BaseGroupsConfig
	Enable              bool          `json:"enable"`
	ImageFeatureHost    string        `json:"feature_host"`
	ImageFeatureTimeout time.Duration `json:"feature_timeout"`
}

//------------------- ImageGroups -------------------//
var _ feature_group.IImageGroups = new(ImageGroups)

type ImageGroups struct {
	baseGroups   *BaseGroups
	config       ImageGroupsConfig
	imagefeature feature.ImageFeature
}

func NewImageGroups(ctx context.Context, baseGroups *BaseGroups, config ImageGroupsConfig) (*ImageGroups, error) {
	s := &ImageGroups{
		baseGroups: baseGroups,
		config:     config,
		imagefeature: feature.NewImageFeature(
			config.ImageFeatureHost,
			config.ImageFeatureTimeout*time.Second,
			config.Sets.Precision*config.Sets.Dimension,
		),
	}
	return s, nil
}

func (s *ImageGroups) Get(ctx context.Context, name proto.GroupName) (feature_group.IImageGroup, error) {
	if len(name) == 0 {
		return nil, errors.New("Invalid Group Name")
	}
	group, err := s.baseGroups.Get(ctx, name)
	if err != nil {
		return nil, err
	}
	baseGroup, ok := group.(*_BaseGroup)
	if !ok {
		return nil, errors.New("baseGroups.Get get invalid *_BaseGroup")
	}
	gs := &ImageGroup{
		baseGroup: baseGroup,
		manager:   s,
	}
	return gs, nil
}

func (s *ImageGroups) New(ctx context.Context, name proto.GroupName, config proto.GroupConfig) (err error) {
	return s.baseGroups.New(ctx, false, name, config)
}

func (s *ImageGroups) All(ctx context.Context) ([]proto.GroupName, error) {
	return s.baseGroups.All(ctx)
}

//------------------- ImageGroup -------------------//
var _ feature_group.IImageGroup = new(ImageGroup)

type ImageGroup struct {
	baseGroup *_BaseGroup
	manager   *ImageGroups
}

func (s *ImageGroup) AddImage(ctx context.Context, images ...proto.Image) (err error) {
	xl := xlog.FromContextSafe(ctx)

	features := make([]proto.Feature, 0)
	for _, i := range images {
		fv, err := s.manager.imagefeature.Image(ctx, i.URI)
		if err != nil {
			xl.Errorf("Parse feature error: %s", err)
			return err
		}
		features = append(features, proto.Feature{
			ID:      i.ID,
			Value:   fv,
			Tag:     i.Tag,
			Desc:    i.Desc,
			HashKey: proto.FeatureHashKey(crc32.ChecksumIEEE([]byte(i.ID))),
		})
	}
	err = s.baseGroup.Add(ctx, false, features...)
	if err != nil {
		xl.Errorf("Group.AddImage error: %s", err)
		return
	}
	return
}

func (s *ImageGroup) UpdateImage(ctx context.Context, images ...proto.Image) (err error) {
	xl := xlog.FromContextSafe(ctx)

	features := make([]proto.Feature, 0)
	for _, i := range images {
		fv, err := s.manager.imagefeature.Image(ctx, i.URI)
		if err != nil {
			xl.Errorf("Parse feature error: %s", err)
			return err
		}
		features = append(features, proto.Feature{
			ID:    i.ID,
			Value: fv,
			Tag:   i.Tag,
			Desc:  i.Desc,
		})
	}
	err = s.baseGroup.Update(ctx, false, features...)
	if err != nil {
		xl.Errorf("Group.UpdateImage error: %s", err)
		return
	}
	return
}

func (s *ImageGroup) SearchImage(ctx context.Context,
	threshold float32, limit int,
	images ...proto.ImageURI,
) (
	[][]feature_group.ImageSearchRespItem,
	error,
) {
	if limit == 0 {
		limit = defaultSearchLimit
	}
	xl := xlog.FromContextSafe(ctx)
	fvs := make([]proto.FeatureValue, 0)
	for index, iURI := range images {
		fv, err := s.manager.imagefeature.Image(ctx, iURI)
		if err != nil {
			return nil, errors.Wrapf(err, "imageFeature failed, index: %v", index)
		}
		fvs = append(fvs, fv)
	}

	searchResult, err := s.baseGroup.Search(ctx, false, threshold, limit, fvs...)
	if err != nil {
		xl.Errorf("Group.Search error: %s", err)
		return nil, err
	}
	ret := make([][]feature_group.ImageSearchRespItem, 0)
	for r, row := range searchResult {
		ret = append(ret, make([]feature_group.ImageSearchRespItem, 0))
		for _, item := range row {
			ret[r] = append(ret[r], feature_group.ImageSearchRespItem{
				ID:    item.Value.ID,
				Score: item.Score,
				Tag:   item.Value.Tag,
				Desc:  item.Value.Desc,
			})
		}
	}
	return ret, nil
}

func (s *ImageGroup) ListImage(ctx context.Context,
	tag proto.FeatureTag, marker string, limit int,
) (
	images []feature_group.ImageListRespItem,
	nextMarker string,
	err error,
) {
	xl := xlog.FromContextSafe(ctx)
	if limit == 0 {
		limit = defaultSearchLimit
	}
	if marker != "" {
		marker, err = decodeMarker(marker)
		if err != nil {
			return nil, "", errors.New("Invalid marker")
		}
	}

	features, nextMarker, err := s.baseGroup.FilterByTag(ctx, tag, marker, limit)
	if err != nil {
		xl.Errorf("Group.ListImage error: %s", err)
		return nil, "", err
	}
	images = make([]feature_group.ImageListRespItem, 0)
	for _, feature := range features {
		images = append(images, feature_group.ImageListRespItem{
			ID:   feature.ID,
			Tag:  feature.Tag,
			Desc: feature.Desc,
		})
	}

	return images, encodeMarker(nextMarker), nil
}

func decodeMarker(marker string) (string, error) {
	s, err := base64.URLEncoding.DecodeString(marker)
	if err != nil {
		return "", err
	}
	return string(s), nil
}

func encodeMarker(pos string) string {
	return base64.URLEncoding.EncodeToString([]byte(pos))
}
