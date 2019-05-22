package feature_group

import (
	"context"
	"testing"

	"gopkg.in/mgo.v2/bson"

	"github.com/stretchr/testify/assert"

	"github.com/qiniu/db/mgoutil.v3"
)

type MockHubInMgoColls struct {
}

func TestHub(t *testing.T) {

	hub, _ := NewHubInMgo(
		&mgoutil.Config{DB: "IG_UT_HUB"},
		&struct {
			Hubs     mgoutil.Collection `coll:"ig_hub_hubs"`
			Features mgoutil.Collection `coll:"ig_hub_features"`
		}{},
	)
	hub.hubs.RemoveAll(bson.M{})
	hub.features.RemoveAll(bson.M{})
	hid, err := hub.New(context.Background(), 4*2, EmptyFeatureVersion)
	assert.NoError(t, err)
	err = hub.Remove(context.Background(), hid)
	assert.NoError(t, err)

	hid, err = hub.New(context.Background(), 4*2, EmptyFeatureVersion)
	assert.NoError(t, err)
	err = hub.Set(context.Background(), hid, "A", []byte{1, 0, 0, 0, 0, 0, 0, 0})
	assert.NoError(t, err)
	fbs, err := hub.All(context.Background(), hid, 4)
	assert.NoError(t, err)
	assert.Equal(t, 1, len(fbs))
	assert.Equal(t, hid, fbs[0].Hid)
	assert.Equal(t, uint64(8), fbs[0].ChunkSize)
	assert.Equal(t, 0, fbs[0].From)
	assert.Equal(t, 1, fbs[0].To)

	fid, err := hub.Find(context.Background(), hid, fbs[0].Ver, 0)
	assert.NoError(t, err)
	assert.Equal(t, FeatureID("A"), fid)

	err = hub.Set(context.Background(), hid, "B", []byte{0, 1, 0, 0, 0, 0, 0, 0})
	err = hub.Set(context.Background(), hid, "C", []byte{0, 0, 1, 0, 0, 0, 0, 0})
	err = hub.Set(context.Background(), hid, "D", []byte{0, 0, 0, 1, 0, 0, 0, 0})
	err = hub.Set(context.Background(), hid, "E", []byte{0, 0, 0, 0, 1, 0, 0, 0})
	err = hub.Del(context.Background(), hid, "B")
	err = hub.Set(context.Background(), hid, "C", []byte{0, 0, 0, 0, 0, 1, 0, 0})

	fbs, err = hub.All(context.Background(), hid, 4)
	assert.NoError(t, err)
	assert.Equal(t, 2, len(fbs))
	assert.Equal(t, 0, fbs[0].From)
	assert.Equal(t, 4, fbs[0].To)
	assert.Equal(t, 4, fbs[1].From)
	assert.Equal(t, 6, fbs[1].To)

	fid, err = hub.Find(context.Background(), hid, fbs[0].Ver, 1)
	assert.NotNil(t, err)
	fid, err = hub.Find(context.Background(), hid, fbs[0].Ver, 2)
	assert.NotNil(t, err)
	fid, _ = hub.Find(context.Background(), hid, fbs[1].Ver, 5)
	assert.Equal(t, FeatureID("C"), fid)

	iter, err := hub.Fetch(context.Background(), hid, fbs[0].Ver, 0, 4)
	assert.NoError(t, err)
	bs, ok := iter.Next(context.Background())
	assert.True(t, ok)
	assert.Equal(t, byte(1), bs[0])
	bs, ok = iter.Next(context.Background())
	assert.True(t, ok)
	assert.Equal(t, byte(0), bs[1])
	bs, ok = iter.Next(context.Background())
	assert.True(t, ok)
	assert.Equal(t, byte(0), bs[2])
	bs, ok = iter.Next(context.Background())
	assert.True(t, ok)
	assert.Equal(t, byte(1), bs[3])
	_, ok = iter.Next(context.Background())
	assert.False(t, ok)

	iter, err = hub.Fetch(context.Background(), hid, fbs[1].Ver, 4, 6)
	assert.NoError(t, err)
	bs, ok = iter.Next(context.Background())
	assert.True(t, ok)
	assert.Equal(t, byte(1), bs[4])
	bs, ok = iter.Next(context.Background())
	assert.True(t, ok)
	assert.Equal(t, byte(1), bs[5])
	_, ok = iter.Next(context.Background())
	assert.False(t, ok)
}
