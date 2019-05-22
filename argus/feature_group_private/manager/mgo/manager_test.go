package mgo

import (
	"context"
	"encoding/json"
	"testing"

	"github.com/qiniu/db/mgoutil.v3"
	"github.com/stretchr/testify/assert"
	mgo_org "gopkg.in/mgo.v2"
	"gopkg.in/mgo.v2/bson"
	"qiniu.com/argus/feature_group_private/proto"
)

const (
	_MGO_HOST = "localhost"
	_MGO_DB   = "test"
)

func dropDatabase(t *testing.T) {
	session, err := mgo_org.Dial(_MGO_HOST)
	if err != nil {
		t.Fatal(err)
	}
	defer session.Close()
	session.SetMode(mgo_org.Monotonic, true)
	session.DB(_MGO_DB).DropDatabase()
}

func iter(ctx context.Context, features ...proto.Feature) error {
	return nil
}

func TestGroupsManager(t *testing.T) {
	a := assert.New(t)
	dropDatabase(t)
	ctx := context.Background()

	groupsConfig := GroupsConfig{
		MgoConfig: mgoutil.Config{
			Host: _MGO_HOST,
			DB:   _MGO_DB,
			Mode: "strong",
		},
		CollSessionPoolLimit: 0,
	}

	groupConfig := proto.GroupConfig{
		Dimension: 2,
		Precision: 2,
		Capacity:  10,
	}

	groupsManager, err := NewGroupsManager(&groupsConfig)
	a.Nil(err)
	a.NotNil(groupsManager)
	groupNames, err := groupsManager.All(ctx)
	a.Nil(err)
	a.Len(groupNames, 0)

	a.Error(groupsManager.New(ctx, "", groupConfig))
	_, err = groupsManager.Get(ctx, "")
	a.Error(err)

	groupsManager.New(ctx, "demo", groupConfig)

	groupNames, err = groupsManager.All(ctx)
	a.Nil(err)
	a.Len(groupNames, 1)

	groupManager, err := groupsManager.Get(ctx, "demo")
	a.Nil(err)
	a.NotNil(groupManager.Config(ctx))
	a.Nil(groupManager.Add(ctx))

	groupCount, err := groupManager.Count(ctx, proto.HashKeyRange{})
	a.Nil(err)
	a.Equal(0, groupCount)

	err = groupManager.Add(ctx, proto.Feature{ID: "1", Tag: "tag", Value: proto.FeatureValue("value"), HashKey: 1})
	a.Nil(err)
	groupCount, err = groupManager.Count(ctx, proto.HashKeyRange{})
	a.Nil(err)
	a.Equal(1, groupCount)
	err = groupManager.Add(ctx, proto.Feature{ID: "2", Tag: "tag", Value: proto.FeatureValue("value"), HashKey: 2})
	a.Nil(err)
	err = groupManager.Add(ctx, proto.Feature{ID: "3", Tag: "tag", Value: proto.FeatureValue("value"), HashKey: 2})
	a.Nil(err)
	err = groupManager.Add(ctx, proto.Feature{ID: "0", Tag: "tag0", Value: proto.FeatureValue("value"), HashKey: 2})
	a.Nil(err)
	groupCount, err = groupManager.Count(ctx, proto.HashKeyRange{})
	a.Equal(4, groupCount)
	groupCount, err = groupManager.Count(ctx, proto.HashKeyRange{0, 2})
	a.Equal(1, groupCount)

	f, err := groupManager.Get(ctx, proto.FeatureID("1"))
	a.Nil(err)
	a.Equal(proto.FeatureID("1"), f.ID)

	existIds, _ := groupManager.Exist(ctx, "1")
	a.Len(existIds, 1)

	existIds, _ = groupManager.Exist(ctx, "1", "2", "3")
	a.Len(existIds, 3)

	existIds, _ = groupManager.Exist(ctx, "1", "4")
	a.Len(existIds, 1)

	features, nextMarker, _ := groupManager.FilterByTag(ctx, "", "", 10)
	a.Len(features, 4)
	a.Empty(nextMarker)

	features, nextMarker, _ = groupManager.FilterByTag(ctx, "tag", "", 10)
	a.Len(features, 3)
	a.Empty(nextMarker)

	features, nextMarker, _ = groupManager.FilterByTag(ctx, "tag", "", 1)
	a.Len(features, 1)
	a.NotEmpty(nextMarker)

	features, nextMarker, _ = groupManager.FilterByTag(ctx, "tag", "0", 10)
	a.Len(features, 3)
	a.Empty(nextMarker)

	features, nextMarker, _ = groupManager.FilterByTag(ctx, "tag", "2", 10)
	a.Len(features, 1)
	a.Empty(nextMarker)

	features, nextMarker, _ = groupManager.FilterByTag(ctx, "tag1", "", 10)
	a.Len(features, 0)
	a.Empty(nextMarker)

	features, nextMarker, _ = groupManager.FilterByTag(ctx, "tag", "a", 10)
	a.Len(features, 0)
	a.Empty(nextMarker)

	tags, nextMarker, err := groupManager.Tags(ctx, "", 10)
	if a.Nil(err) {
		a.Empty(nextMarker)
		a.Len(tags, 2)
	}

	tags, nextMarker, err = groupManager.Tags(ctx, "", 1)
	if a.Nil(err) {
		a.NotEmpty(nextMarker)
		a.Len(tags, 1)
	}
	tags, nextMarker, err = groupManager.Tags(ctx, nextMarker, 10)
	if a.Nil(err) {
		a.Empty(nextMarker)
		a.Len(tags, 1)
	}

	tagCount, err := groupManager.CountTags(ctx)
	if a.Nil(err) {
		a.Equal(2, tagCount)
	}

	err = groupManager.Update(ctx, proto.Feature{ID: "1", Value: proto.FeatureValue("123hh")})
	a.Nil(err)

	groupCount, _ = groupManager.Count(ctx, proto.HashKeyRange{})
	a.Equal(4, groupCount)

	f, err = groupManager.Get(ctx, proto.FeatureID("1"))
	a.Nil(err)
	a.Equal(proto.FeatureValue("123hh"), f.Value)
	a.Equal(proto.FeatureID("1"), f.ID)

	groupManager.Delete(ctx, "1")
	groupCount, _ = groupManager.Count(ctx, proto.HashKeyRange{})
	a.Equal(3, groupCount)

	err = groupManager.Iter(ctx, proto.HashKeyRange{}, iter)
	a.Nil(err)

	groupManager.Destroy(ctx)
	groupNames, _ = groupsManager.All(ctx)
	a.Len(groupNames, 0)

	err = groupsManager.UpsertNode(ctx, proto.Node{"127.0.0.1:1234", 100, proto.NodeStateInitializing})
	a.Nil(err)
	nodes, err := groupsManager.AllNodes(ctx)
	a.Nil(err)
	a.Equal(1, len(nodes))
	a.Equal(proto.NodeAddress("127.0.0.1:1234"), nodes[0].Address)

	a.NoError(groupManager.EnsureHashKey(ctx, mockF))
}

func mockF(proto.FeatureID) (e proto.FeatureHashKey) {
	return
}

func TestGroupsCompatible(t *testing.T) {
	a := assert.New(t)
	dropDatabase(t)
	ctx := context.Background()

	groupsConfig := GroupsConfig{
		MgoConfig: mgoutil.Config{
			Host: _MGO_HOST,
			DB:   _MGO_DB,
			Mode: "strong",
		},
		CollSessionPoolLimit: 0,
	}

	groupConfig := proto.GroupConfig{
		Dimension: 2,
		Precision: 2,
		Capacity:  10,
	}

	collections := _MgoCollections{}
	_, err := mgoutil.Open(&collections, &groupsConfig.MgoConfig)
	a.Nil(err)

	groupsManager, err := NewGroupsManager(&groupsConfig)
	a.Nil(err)
	a.Nil(groupsManager.New(ctx, "compatible", groupConfig))

	col := collections.Features.CopySession()
	feature := bson.M{
		"group": "compatible",
		"id":    "compatible_test_0001",
		"value": []byte{},
		"tag":   "compatible_tag_test_0001",
		"desc":  "compatible_desc_test_0001",
	}
	a.Nil(col.Insert(feature))
	col.CloseSession()
	groupManager, err := groupsManager.Get(ctx, "compatible")
	a.Nil(err)
	f, err := groupManager.Get(ctx, proto.FeatureID("compatible_test_0001"))
	a.Nil(err)
	b, err := json.Marshal(f.Desc)
	a.Nil(err)
	a.Equal(`"compatible_desc_test_0001"`, string(b))

	err = groupManager.Iter(ctx, proto.HashKeyRange{0, 0}, func(ctx context.Context, features ...proto.Feature) error {
		for _, feature := range features {
			b, err := json.Marshal(feature.Desc)
			a.Nil(err)
			a.Equal(`"compatible_desc_test_0001"`, string(b))
		}
		return nil
	})
	a.Nil(err)

	features, _, err := groupManager.FilterByTag(ctx, proto.FeatureTag("compatible_tag_test_0001"), "", 10)
	a.Nil(err)
	for _, feature := range features {
		b, err := json.Marshal(feature.Desc)
		a.Nil(err)
		a.Equal(`"compatible_desc_test_0001"`, string(b))
	}
}
