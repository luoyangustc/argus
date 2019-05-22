package client

import (
	"context"
	"encoding/json"
	"testing"

	"github.com/stretchr/testify/assert"
	"qiniu.com/argus/ccp/manager/proto"
	"qiniu.com/argus/ccp/manager/proto/kodo"
)

func TestPfopClient(t *testing.T) {

	innerCfg := NewInnerConfig(&Saver{}, "")
	pr := NewPfopRules(innerCfg, nil, "")

	_, err := pr.genSetReq(context.Background(), &proto.Rule{})
	assert.NoError(t, err)

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

	_, err = pr.genSetReq(context.Background(), &proto.Rule{
		Source: srcRaw,
	})
	assert.NoError(t, err)

	kodoAction := kodo.KodoAction{}
	srcAct, _ := json.Marshal(kodoAction)

	rl2 := proto.Rule{
		Type:   proto.TYPE_STREAM,
		Source: srcRaw,
		Action: srcAct,
	}
	rl2.Image.IsOn = true
	rl2.Video.IsOn = true
	err = pr.FillPfopName(context.Background(), &rl2)
	assert.NoError(t, err)

	imgreq := _PfopRulesSetReq{}
	err = pr.fillReqByMimeType(context.Background(), &imgreq, &rl2, MT_IMAGE)
	assert.NoError(t, err)

	vidreq := _PfopRulesSetReq{}
	err = pr.fillReqByMimeType(context.Background(), &vidreq, &rl2, MT_VIDEO)
	assert.NoError(t, err)

	mpr := MockPfopRules{}
	err = mpr.Set(context.Background(), &rl2)
	assert.NoError(t, err)

	err = mpr.Del(context.Background(), &rl2)
	assert.NoError(t, err)

	err = mpr.FillPfopName(context.Background(), &rl2)
	assert.NoError(t, err)

}
