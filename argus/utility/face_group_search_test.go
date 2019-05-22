package utility

import (
	"context"
	"testing"

	"github.com/qiniu/db/mgoutil.v3"

	"github.com/stretchr/testify/assert"
)

func TestFaceGroupInDB(t *testing.T) {

	mgoConf := mgoutil.Config{
		Host: "mongodb://127.0.0.1:27017",
		DB:   "faceGroupSearch",
	}

	faceGroupManager, err := NewFaceGroupManagerInDB(&mgoConf)
	assert.NoError(t, err)

	var (
		uid uint32 = 1111
		gid        = "foo1"
		ID         = "123"
		ctx        = context.Background()
	)

	fgroup, err := faceGroupManager.New(ctx, uid, gid)
	assert.NoError(t, err)

	fgroup.Del(ctx, []string{ID})

	var fid string
	{
		_, err := fgroup.Add(ctx, []_FaceItem{
			_FaceItem{
				ID:      ID,
				Name:    "abc",
				Feature: []byte("abcdefg"),
			},
		})
		assert.NoError(t, err)
		items, err := fgroup.All(ctx)
		assert.NoError(t, err)
		assert.Equal(t, 1, len(items))
		assert.Equal(t, "abc", items[0].Name)
		assert.Equal(t, "abcdefg", string(items[0].Feature))
		fid = items[0].ID
	}
	{
		ctx := context.Background()
		assert.NoError(t, fgroup.Del(ctx, []string{fid}))
		items, err := fgroup.All(ctx)
		assert.NoError(t, err)
		assert.Equal(t, 0, len(items))
	}
	fgroup.Del(ctx, []string{ID})
}

func TestFaceGroupManagerInDB(t *testing.T) {

	mgoConf := mgoutil.Config{
		Host: "mongodb://127.0.0.1:27017",
		DB:   "faceGroupSearch",
	}

	faceGroupManager, err := NewFaceGroupManagerInDB(&mgoConf)
	assert.NoError(t, err)

	var (
		uid uint32 = 1121
		gid        = "foo1"
	)

	{
		ctx := context.Background()
		g, err := faceGroupManager.New(ctx, uid, gid)
		assert.NoError(t, err)
		items, err := g.All(ctx)
		assert.NoError(t, err)
		assert.Equal(t, 0, len(items))
	}
	{
		ctx := context.Background()
		g, err := faceGroupManager.Get(ctx, uid, gid)
		assert.NoError(t, err)
		items, err := g.All(ctx)
		assert.NoError(t, err)
		assert.Equal(t, 0, len(items))
	}
	{
		ctx := context.Background()
		assert.NoError(t, faceGroupManager.Remove(ctx, uid, gid))
		_, err := faceGroupManager.Get(ctx, uid, gid)
		assert.Error(t, err)
	}
}
