package client

import (
	"context"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestSaveBack(t *testing.T) {

	ctx := context.Background()
	innerCfg := NewInnerConfig(&Saver{}, "")
	sb := NewSaveBack(innerCfg, "", nil, nil)
	kc := sb.GetKodoInfo(ctx)
	assert.Nil(t, kc)

	kodo := sb.GetKodoInfo(ctx)
	assert.Nil(t, kodo)

	msb := MockSaveBack{}
	err := msb.Save(ctx, nil, nil)
	assert.NoError(t, err)

	uid, _, _, _, _ := msb.GetInnerBucketInfo(ctx)
	assert.Equal(t, uid, uint32(0))

	kodo2 := msb.GetKodoInfo(ctx)
	assert.Nil(t, kodo2)

}
