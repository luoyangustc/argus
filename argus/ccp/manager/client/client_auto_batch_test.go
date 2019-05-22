package client

import (
	"context"
	"encoding/json"
	"testing"

	"github.com/stretchr/testify/assert"
	"qiniu.com/argus/ccp/manager/proto"
	"qiniu.com/argus/ccp/manager/proto/kodo"
)

func TestAutoClient(t *testing.T) {

	innerCfg := NewInnerConfig(&Saver{}, "")
	bj := NewBJobs(innerCfg, "")
	_, err := bj.genNewReq(context.Background(), &proto.Rule{})
	assert.Error(t, err)

	pfx := "qqq"
	kodoSrc := kodo.KodoSrc{
		Buckets: []struct {
			Bucket string  `json:"bucket"`
			Prefix *string `json:"prefix,omitempty"`
		}{
			struct {
				Bucket string  `json:"bucket"`
				Prefix *string `json:"prefix,omitempty"`
			}{
				Bucket: "argus-bcp-test",
				Prefix: &pfx,
			},
		},
	}
	srcRaw, _ := json.Marshal(kodoSrc)

	_, err = bj.genNewReq(context.Background(), &proto.Rule{
		Source: srcRaw,
	})
	assert.NoError(t, err)

	err = bj.Cancel(context.Background(), 123, "")
	assert.NoError(t, err)
}
