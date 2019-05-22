package mgo

import (
	"context"
	"encoding/json"

	"gopkg.in/mgo.v2"
	"gopkg.in/mgo.v2/bson"

	"github.com/pkg/errors"
	"github.com/qiniu/db/mgoutil.v3"

	"qiniu.com/argus/feature_group_private/manager"
	"qiniu.com/argus/feature_group_private/proto"
)

const (
	defaultCollSessionPoolLimit = 100
	defaultIterCursor           = 256
)

type GroupsConfig struct {
	MgoConfig            mgoutil.Config `json:"mgo_config"`
	CollSessionPoolLimit int            `json:"coll_session_pool_limit"`
}

type _MgoCollections struct {
	Groups   mgoutil.Collection `coll:"groups"`
	Features mgoutil.Collection `coll:"features"`
	Nodes    mgoutil.Collection `coll:"nodes"`
}

var _ manager.Groups = new(Groups)
var _ manager.Group = new(Group)

type Group struct {
	Name         proto.GroupName
	config       proto.GroupConfig
	manager      *Groups
	featuresColl *mgoutil.Collection
}

type Groups struct {
	*GroupsConfig
	groupsColl   *mgoutil.Collection
	featuresColl *mgoutil.Collection
	nodeColl     *mgoutil.Collection
}

type _GroupInMgo struct {
	proto.GroupConfig `bson:"group_config"`
	Name              proto.GroupName `bson:"name"`
}

func NewGroupsManager(cfg *GroupsConfig) (manager.Groups, error) {
	collections := _MgoCollections{}
	mgoSession, err := mgoutil.Open(&collections, &cfg.MgoConfig)
	if err != nil {
		return nil, err
	}

	if cfg.CollSessionPoolLimit == 0 {
		cfg.CollSessionPoolLimit = defaultCollSessionPoolLimit
	}

	mgoSession.SetPoolLimit(cfg.CollSessionPoolLimit)

	// ensure index
	if err = collections.Groups.EnsureIndex(mgo.Index{Key: []string{"name"}, Unique: true}); err != nil {
		return nil, errors.Wrapf(err, "fail ensure groups collections index name")
	}
	if err = collections.Features.EnsureIndex(mgo.Index{Key: []string{"group", "id"}, Unique: true}); err != nil {
		return nil, errors.Wrapf(err, "fail ensure features collections index group,id")
	}
	if err = collections.Features.EnsureIndex(mgo.Index{Key: []string{"group", "hash_key"}, Unique: false}); err != nil {
		return nil, errors.Wrapf(err, "fail ensure features collections index group,hash_key")
	}
	if err = collections.Features.EnsureIndex(mgo.Index{Key: []string{"group", "tag"}, Unique: false}); err != nil {
		return nil, errors.Wrapf(err, "fail ensure features collections index group,tag")
	}

	if err = collections.Nodes.EnsureIndex(mgo.Index{Key: []string{"address"}, Unique: true}); err != nil {
		return nil, errors.Wrapf(err, "fail ensure nodes collections index address")
	}

	gsm := &Groups{
		GroupsConfig: cfg,
		groupsColl:   &collections.Groups,
		featuresColl: &collections.Features,
		nodeColl:     &collections.Nodes}
	return gsm, nil
}

// Groups
func (m *Groups) New(ctx context.Context, name proto.GroupName, groupConfig proto.GroupConfig) error {
	if name == "" || groupConfig.Capacity <= 0 {
		return manager.ErrInvalidGroupParams
	}
	col := m.groupsColl.CopySession()
	defer col.CloseSession()
	if err := col.Insert(_GroupInMgo{Name: name,
		GroupConfig: groupConfig,
	}); err != nil {
		return errors.Wrapf(err, "fail to insert group")
	}
	return nil
}

func (m *Groups) Get(ctx context.Context, name proto.GroupName) (manager.Group, error) {
	if name == "" {
		return nil, manager.ErrInvalidGroupParams
	}
	var group _GroupInMgo
	col := m.groupsColl.CopySession()
	defer col.CloseSession()
	if err := col.Find(bson.M{"name": name}).One(&group); err != nil {
		if err != mgo.ErrNotFound {
			return nil, errors.Wrapf(err, "fail to search group")
		}
		return nil, manager.ErrGroupNotExist
	}
	g := &Group{Name: name, config: group.GroupConfig, manager: m, featuresColl: m.featuresColl}
	return g, nil
}

func (m *Groups) All(ctx context.Context) (groupNames []proto.GroupName, err error) {
	groups := []_GroupInMgo{}
	col := m.groupsColl.CopySession()
	defer col.CloseSession()
	if err = col.Find(nil).All(&groups); err != nil {
		return nil, errors.Wrapf(err, "fail to search group")
	}
	for _, g := range groups {
		groupNames = append(groupNames, g.Name)
	}
	return
}

func (m *Groups) AllNodes(ctx context.Context) (nodes []proto.Node, err error) {
	col := m.nodeColl.CopySession()
	defer col.CloseSession()
	if err = col.Find(nil).Sort("address").All(&nodes); err != nil {
		return nil, errors.Wrapf(err, "fail to search node")
	}
	return
}

func (m *Groups) UpsertNode(ctx context.Context, node proto.Node) (err error) {
	col := m.nodeColl.CopySession()
	defer col.CloseSession()

	if _, err = col.Upsert(bson.M{"address": node.Address}, node); err != nil {
		return errors.Wrapf(err, "fail to upsert node")
	}
	return
}

// Group

func compatibleDesc(src json.RawMessage) (ret json.RawMessage) {
	ret = src
	if _, e := json.Marshal(src); e != nil {
		desc := make(json.RawMessage, len(src)+2)
		copy(desc[1:len(src)+1], src)
		desc[0], desc[len(src)+1] = 34, 34
		if _, e = json.Marshal(desc); e == nil {
			ret = desc
		}
	}
	return ret
}

func (m *Group) Config(ctx context.Context) proto.GroupConfig {
	return m.config
}
func (m *Group) Add(ctx context.Context, features ...proto.Feature) (err error) {
	if len(features) == 0 {
		return
	}
	var (
		ids []string
		ffs []interface{}
	)
	for _, feature := range features {
		ids = append(ids, string(feature.ID))
		feature.Group = m.Name
		ffs = append(ffs, feature)
	}
	col := m.featuresColl.CopySession()
	defer col.CloseSession()
	if err = col.Find(bson.M{"group": m.Name, "id": bson.M{"$in": ids}}).One(nil); err == nil {
		return manager.ErrFeatureExist
	}

	if err = col.Insert(ffs...); err != nil {
		return errors.Wrapf(err, "fail to insert feature")
	}
	return
}

func (m *Group) Delete(ctx context.Context, ids ...proto.FeatureID) (err error) {
	col := m.featuresColl.CopySession()
	defer col.CloseSession()
	if _, err = col.RemoveAll(bson.M{"group": m.Name, "id": bson.M{"$in": ids}}); err != nil {
		if err != mgo.ErrNotFound {
			return errors.Wrapf(err, "fail to delete feature")
		}
		return manager.ErrFeatureNotExist
	}
	return
}

func (m *Group) Get(ctx context.Context, id proto.FeatureID) (feature proto.Feature, err error) {
	col := m.featuresColl.CopySession()
	defer col.CloseSession()
	if err = col.Find(bson.M{"group": m.Name, "id": id}).One(&feature); err != nil {
		if err != mgo.ErrNotFound {
			err = errors.Wrapf(err, "fail to get feature")
			return
		}
		err = manager.ErrFeatureNotExist
		return
	}
	// https://jira.qiniu.io/browse/ATLAB-7888
	feature.Desc = compatibleDesc(feature.Desc)
	return
}

func (m *Group) Update(ctx context.Context, features ...proto.Feature) (err error) {
	col := m.featuresColl.CopySession()
	defer col.CloseSession()
	for _, feature := range features {
		if err = col.Update(
			bson.M{"group": m.Name, "id": feature.ID},
			bson.M{"$set": bson.M{
				"value":        feature.Value,
				"tag":          feature.Tag,
				"desc":         feature.Desc,
				"hash_key":     feature.HashKey,
				"bounding_box": feature.BoundingBox}}); err != nil {
			if err != mgo.ErrNotFound {
				return errors.Wrapf(err, "fail to update feature")
			}
			return manager.ErrFeatureNotExist
		}
	}
	return
}

func (m *Group) Count(ctx context.Context, rg proto.HashKeyRange) (count int, err error) {
	col := m.featuresColl.CopySession()
	defer col.CloseSession()
	query := bson.M{"group": m.Name}
	if rg[0] >= 0 && rg[1] > 0 {
		query["hash_key"] = bson.M{"$exists": true, "$gte": rg[0], "$lt": rg[1]}
	}
	if count, err = col.Find(query).Count(); err != nil {
		err = errors.Wrapf(err, "fail to count feature")
		return
	}
	return
}

func (m *Group) CountTags(ctx context.Context) (int, error) {
	col := m.featuresColl.CopySession()
	defer col.CloseSession()
	var result struct {
		Count int `bson:"count"`
	}
	if err := col.Pipe([]bson.M{
		{"$match": bson.M{"group": m.Name}},
		{"$group": bson.M{"_id": "$tag"}},
		{"$group": bson.M{"_id": "count", "count": bson.M{"$sum": 1}}},
	}).One(&result); err != nil {
		if err != mgo.ErrNotFound {
			return 0, errors.Wrapf(err, "fail to count tags")
		}
	}
	return result.Count, nil
}

func (m *Group) CountWithoutHashKey(ctx context.Context) (count int, err error) {
	col := m.featuresColl.CopySession()
	defer col.CloseSession()
	query := bson.M{"group": m.Name, "hash_key": bson.M{"$exists": false}}
	if count, err = col.Find(query).Count(); err != nil {
		err = errors.Wrapf(err, "fail to count feature")
		return
	}
	return
}

func (m *Group) EnsureHashKey(ctx context.Context, f func(proto.FeatureID) proto.FeatureHashKey) (err error) {
	col := m.featuresColl.CopySession()
	defer col.CloseSession()
	iter := col.Find(bson.M{"group": m.Name, "hash_key": bson.M{"$exists": false}}).Iter()
	defer iter.Close()
	var feature proto.Feature
	for iter.Next(&feature) {
		feature.HashKey = f(feature.ID)
		if err = col.Update(bson.M{"group": m.Name, "id": feature.ID}, bson.M{"$set": bson.M{"hash_key": feature.HashKey}}); err != nil {
			if err != mgo.ErrNotFound {
				return errors.Wrapf(err, "fail to update feature")
			}
			return manager.ErrFeatureNotExist
		}
	}
	if err = iter.Close(); err != nil {
		err = errors.Wrapf(err, "fail to close db iter")
		return
	}
	return
}

func (m *Group) Iter(ctx context.Context, rg proto.HashKeyRange, f func(context.Context, ...proto.Feature) error) (err error) {
	col := m.featuresColl.CopySession()
	defer col.CloseSession()
	query := bson.M{"group": m.Name}

	if rg[0] >= 0 && rg[1] > 0 {
		query["hash_key"] = bson.M{"$exists": true, "$gte": rg[0], "$lt": rg[1]}
	}

	iter := col.Find(query).Iter()
	var (
		feature proto.Feature
		reqs    = make([]proto.Feature, defaultIterCursor)
		index   int
	)
	defer iter.Close()
	for iter.Next(&feature) {
		// https: //jira.qiniu.io/browse/ATLAB-7888
		feature.Desc = compatibleDesc(feature.Desc)
		reqs[index] = feature
		index++
		if index >= defaultIterCursor {
			if err = f(ctx, reqs...); err != nil {
				return
			}
			index = 0
		}
	}
	if index > 0 {
		if err = f(ctx, reqs[:index]...); err != nil {
			return
		}
	}
	if err = iter.Close(); err != nil {
		err = errors.Wrapf(err, "fail to close db iter")
		return
	}
	return
}

func (m *Group) Destroy(ctx context.Context) (err error) {
	col := m.featuresColl.CopySession()
	if _, err = col.RemoveAll(bson.M{"group": m.Name}); err != nil {
		col.CloseSession()
		if err != mgo.ErrNotFound {
			return errors.Wrapf(err, "fail to remove feature")
		}
	}
	col.CloseSession()
	col2 := m.manager.groupsColl.CopySession()
	defer col2.CloseSession()
	_, err = col2.RemoveAll(bson.M{"name": m.Name})
	if err != nil {
		return errors.Wrapf(err, "fail to remove group")
	}
	return
}

func (m *Group) Tags(ctx context.Context, marker string, limit int) (tags []proto.GroupTagInfo, nextMarker string, err error) {
	col := m.featuresColl.CopySession()
	defer col.CloseSession()
	tags = make([]proto.GroupTagInfo, 0)
	if err := col.Pipe([]bson.M{
		{"$match": bson.M{"group": m.Name}},
		{"$group": bson.M{"_id": "$tag", "count": bson.M{"$sum": 1}}},
		{"$sort": bson.M{"_id": 1}},
		{"$match": bson.M{"_id": bson.M{"$gt": marker}}},
		// NOTE: elvis, 这里取出limit+1个是为了确认之后已经没有了, nextMarker须为空
		{"$limit": limit + 1},
	}).All(&tags); err != nil {
		if err != mgo.ErrNotFound {
			return nil, "", errors.Wrapf(err, "fail to search feature tags")
		}
	}
	if len(tags) > limit {
		nextMarker = string(tags[limit-1].Name)
		tags = tags[:limit]
	}
	return
}

func (m *Group) FilterByTag(ctx context.Context, tag proto.FeatureTag, marker string, limit int) (features []proto.Feature, nextMarker string, err error) {
	query := bson.M{"group": m.Name, "id": bson.M{"$gt": marker}}
	if tag != "" {
		query["tag"] = tag
	}
	col := m.featuresColl.CopySession()
	defer col.CloseSession()
	// NOTE: elvis, 这里取出limit+1个是为了确认之后已经没有了, nextMarker须为空
	if err = col.Find(query).Sort("id").Limit(limit + 1).All(&features); err != nil {
		if err != mgo.ErrNotFound {
			return nil, "", errors.Wrapf(err, "fail to search feature")
		}
	}
	if len(features) > limit {
		nextMarker = string(features[limit-1].ID)
		features = features[:limit]
	}
	// https: //jira.qiniu.io/browse/ATLAB-7888
	for i, _ := range features {
		features[i].Desc = compatibleDesc(features[i].Desc)
	}
	return
}

func (m *Group) Exist(ctx context.Context, ids ...proto.FeatureID) (existIds []proto.FeatureID, err error) {
	var existFeatures []struct {
		ID proto.FeatureID `bson:"id"`
	}
	existIds = make([]proto.FeatureID, 0)
	col := m.featuresColl.CopySession()
	defer col.CloseSession()
	if err = col.Find(bson.M{"group": m.Name, "id": bson.M{"$in": ids}}).Select(bson.M{"id": 1}).All(&existFeatures); err != nil {
		if err != mgo.ErrNotFound {
			return nil, errors.Wrapf(err, "fail to search feature")
		}
	}
	for _, f := range existFeatures {
		existIds = append(existIds, f.ID)
	}
	return
}
