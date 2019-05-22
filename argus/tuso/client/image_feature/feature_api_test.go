package image_feature

import (
	"context"
	"os"
	"testing"

	"github.com/stretchr/testify/assert"
	"qiniu.com/argus/tuso/proto"
)

func TestFeatureApi(t *testing.T) {
	if os.Getenv("TEST_FEATURE_API") == "" {
		t.SkipNow()
	}
	ctx := context.Background()
	f := NewFeatureApi(FeatureApiConfig{Host: "http://ava-serving-gate.cs.cg.dora-internal.qiniu.io:5001"})
	resp, err := f.PostEvalFeature(ctx, proto.PostEvalFeatureReq{Image: proto.Image{Bucket: "vance-test", Key: "image.png", Uid: 1380531519}})
	a := assert.New(t)
	a.Nil(err)
	a.Equal(len(resp.Feature), proto.FeatureSize)
	// TODO: md5 support
	return
}
