package client

import (
	"context"
	"testing"

	"github.com/stretchr/testify/assert"
	"qiniu.com/argus/ccp/manager/proto"
)

func TestCallback(t *testing.T) {

	ctx := context.Background()
	nc := NewNotifyCallback()

	err := nc.PostNotifyMsg(ctx, &proto.Rule{}, nil)
	assert.Error(t, err)

	nturl := ""
	err = nc.PostNotifyMsg(ctx, &proto.Rule{
		NotifyURL: &nturl,
	}, nil)
	assert.Error(t, err)

}
