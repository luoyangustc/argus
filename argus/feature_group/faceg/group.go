package faceg

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"os"

	"github.com/pkg/errors"
	"github.com/qiniu/http/httputil.v1"

	"gopkg.in/mgo.v2"
	"gopkg.in/mgo.v2/bson"

	"github.com/qiniu/db/mgoutil.v3"
	"github.com/qiniu/xlog.v1"

	FG "qiniu.com/argus/feature_group"
	"qiniu.com/argus/utility"
)

type _FaceGroupManager interface {
	Hub(context.Context) FG.Hub
	Get(context.Context, uint32, string) (_FaceGroup, error)
	All(context.Context, uint32) ([]string, error)
	New(context.Context, uint32, string, FG.FeatureVersion) (_FaceGroup, error)
	Update(context.Context, uint32, string, FG.HubID, FG.HubID) error
	Remove(context.Context, uint32, string) error
}

type _FaceItem struct {
	ID          string                `json:"id" bson:"id"`
	Name        string                `json:"name" bson:"name"`
	Backup      string                `json:"-" bson:"backup"`
	BoundingBox utility.FaceDetectBox `json:"bounding_box" bson:"bounding_box"`
	Desc        json.RawMessage       `json:"desc,omitempty" bson:"desc,omitempty"`
}

type _FaceGroup interface {
	Hub(context.Context) (FG.HubID, FG.Hub)
	Get(context.Context, string) (_FaceItem, error)
	Add(context.Context, []_FaceItem, [][]byte) []error
	Del(context.Context, []string) error
	All(context.Context) ([]_FaceItem, error)
	Count(context.Context) (int, error)
	Iter(context.Context) (FaceGroupIter, error)

	CheckByID(context.Context, string) (bool, error)
}

type FaceGroupIter interface {
	Next(ctx context.Context) (*_FaceItem, bool)
	Error() error
	Close()
}

var _ _FaceGroupManager = faceGroupManagerInMgo{}

type faceGroupManagerInMgo struct {
	hub    FG.Hub
	groups *mgoutil.Collection
	faces  *mgoutil.Collection
}

func NewFaceGroupManagerInMgo(mgoConf *mgoutil.Config, hub FG.Hub) (faceGroupManagerInMgo, error) {

	var (
		colls struct {
			Groups mgoutil.Collection `coll:"fg_groups"`
			Faces  mgoutil.Collection `coll:"fg_faces"`
		}
	)
	sess, err := mgoutil.Open(&colls, mgoConf)
	if err != nil {
		return faceGroupManagerInMgo{}, err
	}
	sess.SetPoolLimit(100) // DefaultCollSessionPoolLimit
	err = colls.Groups.EnsureIndex(mgo.Index{Key: []string{"uid", "id"}, Unique: true})
	if err != nil {
		return faceGroupManagerInMgo{}, errors.Wrap(err, "Groups.EnsureIndex uid id")
	}
	err = colls.Faces.EnsureIndex(mgo.Index{Key: []string{"uid", "gid", "id"}, Unique: true})
	if err != nil {
		return faceGroupManagerInMgo{}, errors.Wrap(err, "Groups.EnsureIndex uid gid id")
	}
	err = colls.Faces.EnsureIndex(mgo.Index{Key: []string{"uid", "gid"}})
	if err != nil {
		return faceGroupManagerInMgo{}, errors.Wrap(err, "Groups.EnsureIndex uid gid")
	}

	return faceGroupManagerInMgo{hub: hub, groups: &colls.Groups, faces: &colls.Faces}, nil
}

func (m faceGroupManagerInMgo) Hub(context.Context) FG.Hub { return m.hub }

func (m faceGroupManagerInMgo) Get(ctx context.Context, uid uint32, id string) (_FaceGroup, error) {

	g := m.groups.CopySession()
	defer g.CloseSession()

	type Id struct {
		ID  string `bson:"id"`
		HID string `bson:"hid"`
	}
	var _id Id
	err := g.Find(bson.M{"uid": uid, "id": id}).Select(bson.M{"_id": 0, "id": 1, "hid": 1}).One(&_id)
	if mgo.ErrNotFound == err {
		return nil, httputil.NewError(http.StatusBadRequest, fmt.Sprintf(`group %s not exists`, id))
	}
	if err != nil {
		return nil, err
	}
	return faceGroupInMgo{
		Collection: m.faces, hub: m.hub,
		uid: uid, id: id, hid: FG.HubID(_id.HID)}, nil
}

func (m faceGroupManagerInMgo) All(ctx context.Context, uid uint32) ([]string, error) {

	g := m.groups.CopySession()
	defer g.CloseSession()

	type Id struct {
		ID string `bson:"id"`
	}
	var ids = make([]Id, 0)
	var ret = make([]string, 0)

	err := g.Find(bson.M{"uid": uid}).Select(bson.M{"id": 1}).All(&ids)
	if err != nil {
		return nil, err
	}
	for _, id := range ids {
		ret = append(ret, id.ID)
	}
	return ret, nil
}

func (m faceGroupManagerInMgo) New(ctx context.Context, uid uint32, id string, featureVersion FG.FeatureVersion) (_FaceGroup, error) {

	coll := m.groups.CopySession()
	defer coll.CloseSession()

	hubID, err := m.hub.New(ctx, 512*4, featureVersion) // TODO
	if err != nil {
		return nil, errors.Wrap(err, "hub.New")
	}

	err = coll.Insert(bson.M{"uid": uid, "id": id, "hid": hubID})
	if err != nil {
		return nil, err
	}
	return faceGroupInMgo{Collection: m.faces, hub: m.hub, uid: uid, id: id, hid: hubID}, nil
}

func (m faceGroupManagerInMgo) Update(ctx context.Context, uid uint32, id string, oldHid FG.HubID, newHid FG.HubID) error {
	coll := m.groups.CopySession()
	defer coll.CloseSession()

	_, err := coll.Find(bson.M{"uid": uid, "id": id, "hid": oldHid}).
		Apply(mgo.Change{
			Update: bson.M{
				"$set": bson.M{
					"hid": newHid,
				},
			},
		}, nil)

	return err
}

func (m faceGroupManagerInMgo) Remove(ctx context.Context, uid uint32, id string) error {

	g := m.groups.CopySession()
	defer g.CloseSession()

	f := m.faces.CopySession()
	defer f.CloseSession()

	xl := xlog.FromContextSafe(ctx)
	_, err := f.RemoveAll(bson.M{"uid": uid, "gid": id})
	if err != nil {
		xl.Errorf("remove those faces with uid %v gid %v error:%v ", uid, id, err)
		return err
	}
	_, err = g.RemoveAll(bson.M{"uid": uid, "id": id})
	if err != nil {
		xl.Errorf("remove group with uid %v gid %v error:%v ", uid, id, err)
		return err
	}
	return nil
}

type faceGroupInMgo struct {
	*mgoutil.Collection
	hub FG.Hub
	uid uint32
	id  string
	hid FG.HubID
}

func (g faceGroupInMgo) Hub(context.Context) (FG.HubID, FG.Hub) { return g.hid, g.hub }

func (g faceGroupInMgo) Get(ctx context.Context, id string) (_FaceItem, error) {
	coll := g.CopySession()
	defer coll.CloseSession()

	var item _FaceItem
	err := coll.Find(bson.M{"uid": g.uid, "gid": g.id, "id": id}).One(&item)
	if mgo.ErrNotFound == err {
		return item, os.ErrNotExist
	}
	return item, err
}

func (g faceGroupInMgo) Add(ctx context.Context, items []_FaceItem, features [][]byte) []error {

	f := g.CopySession()
	defer f.CloseSession()

	_items := make([]interface{}, 0, len(items))
	for _, item := range items {
		_items = append(_items,
			struct {
				UID         uint32                `bson:"uid"`
				Gid         string                `bson:"gid"`
				ID          string                `bson:"id"`
				Name        string                `bson:"name"`
				Backup      string                `bson:"backup"`
				BoundingBox utility.FaceDetectBox `bson:"bounding_box"`
				Desc        json.RawMessage       `bson:"desc,omitempty"`
			}{
				UID:         g.uid,
				Gid:         g.id,
				ID:          item.ID,
				Name:        item.Name,
				Backup:      item.Backup,
				BoundingBox: item.BoundingBox,
				Desc:        item.Desc,
			},
		)
	}

	errs := make([]error, len(_items))
	for i := range _items {
		errs[i] = f.Insert(_items[i])
		if mgo.IsDup(errs[i]) {
			errs[i] = httputil.NewError(http.StatusBadRequest, `id already exists`)
		}
		if errs[i] == nil {
			errs[i] = g.hub.Set(ctx, g.hid, FG.FeatureID(items[i].ID), features[i])
		}
	}
	return errs
}

func (g faceGroupInMgo) Del(ctx context.Context, ids []string) error {

	f := g.CopySession()
	defer f.CloseSession()

	_, err := f.RemoveAll(bson.M{"uid": g.uid, "gid": g.id, "id": bson.M{"$in": ids}})
	return err
}

func (g faceGroupInMgo) All(ctx context.Context) ([]_FaceItem, error) {

	f := g.CopySession()
	defer f.CloseSession()

	items := make([]_FaceItem, 0)

	err := f.
		Find(bson.M{"uid": g.uid, "gid": g.id}).
		Select(bson.M{"_id": 0, "uid": 0, "gid": 0}).
		All(&items)
	return items, err
}

func (g faceGroupInMgo) Count(ctx context.Context) (int, error) {

	f := g.CopySession()
	defer f.CloseSession()

	return f.Find(bson.M{"uid": g.uid, "gid": g.id}).Count()
}

func (g faceGroupInMgo) Iter(ctx context.Context) (FaceGroupIter, error) {
	f := g.CopySession()
	//defer f.CloseSession()

	iter := f.
		Find(bson.M{"uid": g.uid, "gid": g.id}).
		Select(bson.M{"_id": 0, "uid": 0, "gid": 0}).
		Iter()

	return _FaceGroupIter{
		coll: f,
		iter: iter,
	}, nil
}

func (g faceGroupInMgo) CheckByID(ctx context.Context, id string) (bool, error) {
	f := g.CopySession()
	defer f.CloseSession()

	n, err := f.Find(bson.M{"id": id}).Count()
	if err != nil {
		return false, err
	}
	return n >= 1, nil
}

type _FaceGroupIter struct {
	iter *mgo.Iter
	coll mgoutil.Collection
}

func (fi _FaceGroupIter) Next(ctx context.Context) (*_FaceItem, bool) {
	var item _FaceItem
	ok := fi.iter.Next(&item)
	if ok {
		return &item, true
	}
	return nil, false
}

func (fi _FaceGroupIter) Error() error { return fi.iter.Err() }

func (fi _FaceGroupIter) Close() { fi.coll.CloseSession() }
