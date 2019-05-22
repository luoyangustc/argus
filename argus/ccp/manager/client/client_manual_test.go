package client

import (
	"context"
	"testing"

	"github.com/stretchr/testify/assert"
	"qiniu.com/argus/ccp/manager/proto"
)

func TestManualJobs(t *testing.T) {

	innerCfg := NewInnerConfig(&Saver{}, "")

	mj := NewManualJobs(innerCfg, "")

	err := mj.NewSet(context.Background(), &proto.Rule{})
	assert.Error(t, err)

	err = mj.PushItem(context.Background(), nil, nil)
	assert.NoError(t, err)

	//====

	mmj := MockManualJobs{}
	err = mmj.NewSet(context.Background(), &proto.Rule{})
	assert.NoError(t, err)

	err = mmj.PushItem(context.Background(), nil, nil)
	assert.NoError(t, err)

	err = mmj.PushItems(context.Background(), nil, nil)
	assert.NoError(t, err)

}
