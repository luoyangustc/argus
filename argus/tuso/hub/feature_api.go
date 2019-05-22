package hub

import (
	"context"

	"github.com/pkg/errors"
	"qiniu.com/argus/tuso/proto"
)

// TODO: mock it to mock_test.go
type mockImageFeatureAPI struct {
	mode string
}

func (m *mockImageFeatureAPI) PostEvalFeature(ctx context.Context, req proto.PostEvalFeatureReq) (resp *proto.PostEvalFeatureResp, err error) {
	feature := make([]byte, proto.FeatureSize)
	feature[0] = 11
	feature[proto.FeatureSize-1] = 22
	if m.mode == "" {
		return &proto.PostEvalFeatureResp{
			Feature: feature,
			Md5:     "d41d8cd98f00b204e9800998ecf8427e",
		}, nil
	}
	if m.mode == "hasError" {
		if req.Image.Key == "2.jpg" {
			return nil, errors.New("Mock Server Error")
		}
		return &proto.PostEvalFeatureResp{
			Feature: feature,
			Md5:     "d41d8cd98f00b204e9800998ecf8427e",
		}, nil
	}
	panic("bad mode")
}
