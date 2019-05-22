package manager

import (
	"context"
	"fmt"
	"net/http"

	"gopkg.in/mgo.v2/bson"

	"gopkg.in/mgo.v2"

	"github.com/qiniu/db/mgoutil.v3"
	"github.com/qiniu/http/httputil.v1"
	"qiniu.com/argus/AIProjects/tianyan/serving"
)

const (
	defaultCollSessionPoolLimit = 100
	defaultIterCursor           = 256
)

var (
	ErrInvalidGroupParams = httputil.NewError(http.StatusBadRequest, "invalid group params")
	ErrGroupExist         = httputil.NewError(http.StatusBadRequest, "group is already exist")
	ErrGroupNotExist      = httputil.NewError(http.StatusBadRequest, "group is not exist")
	ErrFeatureExist       = httputil.NewError(http.StatusBadRequest, "feature is already exist")
	ErrFeatureNotExist    = httputil.NewError(http.StatusBadRequest, "feature is not exist")
)

type ManagerConfig struct {
	MgoConfig            mgoutil.Config `json:"mgo_config"`
	CollSessionPoolLimit int            `json:"coll_session_pool_limit"`
}

type _Collections struct {
	Groups   mgoutil.Collection `coll:"groups"`
	Features mgoutil.Collection `coll:"features"`
}

type _FeatureManager struct {
	*ManagerConfig
	groupsColl   *mgoutil.Collection
	featuresColl *mgoutil.Collection
}

type Manager interface {
	AddGroup(context.Context, string, int) error
	DeleteGroup(context.Context, string) error
	GetGroup(context.Context, string) (Group, error)
	AllGroups(context.Context) ([]Group, error)

	AddFeatures(context.Context, string, []serving.Feature) error
	DeleteFeatures(context.Context, string, []string) error
	GetFeature(context.Context, string) (*serving.Feature, error)
	UpdateFeature(context.Context, *serving.Feature) error
	CountFeautre(context.Context, string) (int, error)
	IterFeature(context.Context, string, func(context.Context, serving.FSAddReq) error) error
}

// group working state
const (
	GroupUnknown int = iota
	GroupCreated
	GroupInitialized
)

type Group struct {
	ID        bson.ObjectId `bson:"_id"`
	Name      string        `bson:"name"`
	Capacity  int           `bson:"capacity"`
	Dimension int           `bson:"demension"`
	Precision int           `bson:"precision"`
	Version   uint64        `bson:"version"`
	State     int           `bson:"state"`
}

type Feature struct {
	ID    string        `bson:"id"`
	Name  string        `bson:"name,omitempty"`
	Value []byte        `bson:"Value"`
	GID   bson.ObjectId `bson:"group_id"`
}

func NewManager(cfg *ManagerConfig) (*_FeatureManager, error) {
	collections := _Collections{}
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
		return nil, fmt.Errorf("groups collections ensure index name err: %s", err.Error())
	}
	if err = collections.Features.EnsureIndex(mgo.Index{Key: []string{"id"}, Unique: true}); err != nil {
		return nil, fmt.Errorf("features collections ensure index id err: %s", err.Error())
	}
	if err = collections.Features.EnsureIndex(mgo.Index{Key: []string{"group_id"}}); err != nil {
		return nil, fmt.Errorf("features collections ensure index group_id err: %s", err.Error())
	}

	return &_FeatureManager{ManagerConfig: cfg, groupsColl: &collections.Groups, featuresColl: &collections.Features}, nil
}

func (m *_FeatureManager) AddGroup(ctx context.Context, name string, capacity int) (err error) {
	if name == "" || capacity <= 0 {
		return ErrInvalidGroupParams
	}

	// check if the group is exist in db before call AddGroup

	col := m.groupsColl.CopySession()
	defer col.CloseSession()
	if err = col.Insert(Group{ID: bson.NewObjectId(), Name: name, Capacity: capacity, State: GroupInitialized}); err != nil {
		return httputil.NewError(http.StatusInternalServerError, fmt.Sprintf("fail to add group, insert group failed due to db err: %s", err.Error()))
	}
	return
}

func (m *_FeatureManager) UpdateGroupState(ctx context.Context, name string, state int) (err error) {
	if name == "" || state < GroupUnknown {
		return ErrInvalidGroupParams
	}

	col := m.groupsColl.CopySession()
	defer col.CloseSession()
	if err = col.Update(M{"name": name}, M{"$set": M{"state": state}}); err != nil {
		return httputil.NewError(http.StatusInternalServerError, fmt.Sprintf("fail to update group state %d, update group failed due to db err: %s", state, err.Error()))
	}
	return
}

func (m *_FeatureManager) DeleteGroup(ctx context.Context, name string) (err error) {
	if name == "" {
		return ErrInvalidGroupParams
	}
	g, err := m.GetGroup(ctx, name)
	if err != nil {
		return err
	}

	col1 := m.featuresColl.CopySession()
	defer col1.CloseSession()

	if _, err = col1.RemoveAll(M{"group_id": g.ID}); err != nil {
		if err != mgo.ErrNotFound {
			return httputil.NewError(http.StatusInternalServerError, fmt.Sprintf("fail to delete group, remove features failed due to db err: %s", err.Error()))
		}
	}

	col2 := m.groupsColl.CopySession()
	defer col2.CloseSession()
	if err = col2.Remove(M{"name": name}); err != nil {
		if err != mgo.ErrNotFound {
			return httputil.NewError(http.StatusInternalServerError, fmt.Sprintf("fail to delete group, remove group failed due to db err: %s", err.Error()))
		}
		return ErrGroupNotExist
	}

	return
}

func (m *_FeatureManager) GetGroup(ctx context.Context, name string) (group Group, err error) {
	if name == "" {
		err = ErrInvalidGroupParams
		return
	}

	col := m.groupsColl.CopySession()
	defer col.CloseSession()
	if err = col.Find(M{"name": name}).One(&group); err != nil {
		if err != mgo.ErrNotFound {
			err = httputil.NewError(http.StatusInternalServerError, fmt.Sprintf("fail to get group due to db error: %s", err.Error()))
		}
		// group is exist
		err = ErrGroupNotExist
		return
	}
	return
}

func (m *_FeatureManager) AllGroups(ctx context.Context) (groups []Group, err error) {
	col := m.groupsColl.CopySession()
	defer col.CloseSession()
	if err = col.Find(nil).All(&groups); err != nil {
		err = httputil.NewError(http.StatusInternalServerError, fmt.Sprintf("fail to get all groups due to db error: %s", err.Error()))
	}
	return
}

func (m *_FeatureManager) AddFeatures(ctx context.Context, group string, features []serving.Feature) (err error) {

	if len(features) == 0 {
		return
	}

	g, e := m.GetGroup(ctx, group)
	if e != nil {
		return e
	}

	// check if the feature is exist in db
	var (
		ids []string
		ffs []interface{}
	)
	for _, feature := range features {
		ids = append(ids, feature.ID)
		ffs = append(ffs, Feature{GID: g.ID, ID: feature.ID, Name: feature.Name, Value: feature.Value})
	}
	col1 := m.groupsColl.CopySession()
	defer col1.CloseSession()
	if err = col1.Find(M{"id": M{"$in": ids}}).One(nil); err == nil {
		// feature is exist
		return ErrFeatureExist
	}

	if err != mgo.ErrNotFound {
		return httputil.NewError(http.StatusInternalServerError, fmt.Sprintf("fail to add features, check feature if exist failed due to db error: %s", err.Error()))
	}

	col2 := m.featuresColl.CopySession()
	defer col2.CloseSession()
	if err = col2.Insert(ffs...); err != nil {
		return httputil.NewError(http.StatusInternalServerError, fmt.Sprintf("fail to add feature due to db err: %s", err.Error()))
	}
	if err = col1.Update(M{"_id": g.ID}, M{"$inc": M{"version": 1}}); err != nil {
		return httputil.NewError(http.StatusInternalServerError, fmt.Sprintf("add features, fail to update group version due to db err: %s", err.Error()))
	}
	return
}

func (m *_FeatureManager) DeleteFeatures(ctx context.Context, group string, ids []string) (err error) {

	g, e := m.GetGroup(ctx, group)
	if e != nil {
		return e
	}

	col1 := m.featuresColl.CopySession()
	defer col1.CloseSession()
	if _, err = col1.RemoveAll(M{"id": M{"$in": ids}}); err != nil {
		if err != mgo.ErrNotFound {
			return fmt.Errorf("fail to delete feature due to db err: %s", err.Error())
		}
		return ErrFeatureNotExist
	}
	col2 := m.groupsColl.CopySession()
	defer col2.CloseSession()
	if err = col2.Update(M{"_id": g.ID}, M{"$inc": M{"version": 1}}); err != nil {
		return httputil.NewError(http.StatusInternalServerError, fmt.Sprintf("delete features, fail to update group version due to db err: %s", err.Error()))
	}
	return
}

func (m *_FeatureManager) GetFeature(ctx context.Context, id string) (feature *serving.Feature, err error) {
	var f Feature
	col := m.featuresColl.CopySession()
	defer col.CloseSession()
	if err = col.Find(M{"id": id}).One(&f); err != nil {
		if err != mgo.ErrNotFound {
			err = fmt.Errorf("fail to get feature due to db err: %s", err.Error())
			return
		}
		err = ErrFeatureNotExist
		return
	}
	feature = &serving.Feature{
		Name:  f.Name,
		ID:    f.ID,
		Value: f.Value,
	}
	return
}

func (m *_FeatureManager) UpdateFeature(ctx context.Context, feature *serving.Feature) (err error) {
	col := m.featuresColl.CopySession()
	defer col.CloseSession()
	if err = col.Update(M{"id": feature.Value}, M{"$set": M{"value": feature.Value, "name": feature.Name}}); err != nil {
		if err != mgo.ErrNotFound {
			return fmt.Errorf("fail to update feature due to db err: %s", err.Error())

		}
		return ErrFeatureNotExist
	}
	return
}

func (m *_FeatureManager) CountFeautre(ctx context.Context, group string) (count int, err error) {
	s, e := m.GetGroup(ctx, group)
	if e != nil {
		err = e
		return
	}
	col := m.featuresColl.CopySession()
	defer col.CloseSession()
	if count, err = col.Find(M{"set_id": s.ID}).Count(); err != nil {
		err = fmt.Errorf("fail to get feature due to db err: %s", err.Error())
		return
	}
	return
}

func (m *_FeatureManager) IterFeature(ctx context.Context, group string, f func(context.Context, serving.FSAddReq) error) (err error) {
	s, e := m.GetGroup(ctx, group)
	if e != nil {
		err = e
		return
	}
	col := m.featuresColl.CopySession()
	defer col.CloseSession()
	iter := col.Find(M{"group_id": s.ID}).Iter()
	var (
		feature Feature
		req     = serving.FSAddReq{Name: group}
	)
	for iter.Next(&feature) {
		req.Features = append(req.Features, serving.Feature{ID: feature.ID, Value: feature.Value})
		if len(req.Features) == defaultIterCursor {
			if err = f(ctx, req); err != nil {
				return
			}
			req.Features = make([]serving.Feature, 0)
		}
	}
	if len(req.Features) > 0 {
		if err = f(ctx, req); err != nil {
			return
		}
	}
	if err = iter.Close(); err != nil {
		err = fmt.Errorf("fail to close db iter err: %s", err.Error())
		return
	}
	return
}
