package imageg

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"os"

	"github.com/pkg/errors"
	httputil "qiniupkg.com/http/httputil.v2"

	"gopkg.in/mgo.v2"
	"gopkg.in/mgo.v2/bson"

	"github.com/qiniu/db/mgoutil.v3"
	"github.com/qiniu/xlog.v1"

	FG "qiniu.com/argus/feature_group"
)

type _ImageGroupManager interface {
	Hub(context.Context) FG.Hub
	Get(context.Context, uint32, string) (_ImageGroup, error)
	All(context.Context, uint32) ([]string, error)
	New(context.Context, uint32, string, FG.FeatureVersion) (_ImageGroup, error)
	Update(context.Context, uint32, string, FG.HubID, FG.HubID) error
	Remove(context.Context, uint32, string) error
}

type _ImageItem struct {
	ID     string          `json:"id" bson:"id"`
	Label  string          `json:"label" bson:"label"`
	Etag   string          `json:"etag" bson:"etag"`
	URI    string          `json:"uri" bson:"uri"`
	Backup string          `json:"-" bson:"backup"`
	Desc   json.RawMessage `json:"desc,omitempty" bson:"desc,omitempty"`
}

type _ImageGroup interface {
	Hub(context.Context) (FG.HubID, FG.Hub)
	Get(context.Context, string) (_ImageItem, error)
	Add(context.Context, []_ImageItem, [][]byte) []error
	Del(context.Context, []string) error
	All(context.Context) ([]_ImageItem, error)
	Count(context.Context) (int, error)
	Iter(context.Context) (ImageGroupIter, error)
}

type ImageGroupIter interface {
	Next(ctx context.Context) (*_ImageItem, bool)
	Error() error
	Close()
}

var _ _ImageGroupManager = imageGroupManagerInMgo{}

type imageGroupManagerInMgo struct {
	hub    FG.Hub
	groups *mgoutil.Collection
	images *mgoutil.Collection
}

func NewImageGroupManagerInMgo(mgoConf *mgoutil.Config, hub FG.Hub) (imageGroupManagerInMgo, error) {

	var (
		colls = struct {
			Groups mgoutil.Collection `coll:"ig_groups"`
			Images mgoutil.Collection `coll:"ig_images"`
		}{}
	)
	sess, err := mgoutil.Open(&colls, mgoConf)
	if err != nil {
		return imageGroupManagerInMgo{}, err
	}
	// sess.Close()
	sess.SetPoolLimit(100) // DefaultCollSessionPoolLimit
	err = colls.Groups.EnsureIndex(mgo.Index{Key: []string{"uid", "id"}, Unique: true})
	if err != nil {
		return imageGroupManagerInMgo{}, errors.Wrap(err, "Groups.EnsureIndex uid id")
	}
	err = colls.Images.EnsureIndex(mgo.Index{Key: []string{"uid", "gid", "id"}, Unique: true})
	if err != nil {
		return imageGroupManagerInMgo{}, errors.Wrap(err, "Groups.EnsureIndex uid git id")
	}
	err = colls.Images.EnsureIndex(mgo.Index{Key: []string{"uid", "gid"}})
	if err != nil {
		return imageGroupManagerInMgo{}, errors.Wrap(err, "Groups.EnsureIndex uid gid")
	}

	return imageGroupManagerInMgo{hub: hub, groups: &colls.Groups, images: &colls.Images}, nil
}

func (m imageGroupManagerInMgo) Hub(context.Context) FG.Hub { return m.hub }

func (m imageGroupManagerInMgo) Get(ctx context.Context, uid uint32, id string) (_ImageGroup, error) {

	coll := m.groups.CopySession()
	defer coll.CloseSession()

	var _id = struct {
		ID  string `bson:"id"`
		HID string `bson:"hid"`
	}{}
	err := coll.Find(bson.M{"uid": uid, "id": id}).Select(bson.M{"_id": 0, "id": 1, "hid": 1}).One(&_id)
	if mgo.ErrNotFound == err {
		return nil, httputil.NewError(http.StatusBadRequest, fmt.Sprintf(`group %s not exists`, id))
	}
	if err != nil {
		return nil, err
	}
	return imageGroupInMgo{
		Collection: m.images, hub: m.hub,
		uid: uid, id: id, hid: FG.HubID(_id.HID)}, nil
}

func (m imageGroupManagerInMgo) All(ctx context.Context, uid uint32) ([]string, error) {

	g := m.groups.CopySession()
	defer g.CloseSession()

	var ids = make(
		[]struct {
			ID string `bson:"id"`
		}, 0)
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

func (m imageGroupManagerInMgo) New(ctx context.Context, uid uint32, id string, featureVersion FG.FeatureVersion) (_ImageGroup, error) {

	coll := m.groups.CopySession()
	defer coll.CloseSession()

	hubID, err := m.hub.New(ctx, 4096*4, featureVersion)
	if err != nil {
		return nil, errors.Wrapf(err, "hub.New %v", hubID)
	}

	err = coll.Insert(bson.M{"uid": uid, "id": id, "hid": hubID})
	if err != nil {
		return nil, err
	}
	return imageGroupInMgo{Collection: m.images, hub: m.hub, uid: uid, id: id, hid: hubID}, nil
}

func (m imageGroupManagerInMgo) Update(ctx context.Context, uid uint32, id string, oldHid FG.HubID, newHid FG.HubID) error {
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

func (m imageGroupManagerInMgo) Remove(ctx context.Context, uid uint32, id string) error {

	gColl := m.groups.CopySession()
	defer gColl.CloseSession()

	coll := m.images.CopySession()
	defer coll.CloseSession()

	xl := xlog.FromContextSafe(ctx)
	_, err := coll.RemoveAll(bson.M{"uid": uid, "gid": id})
	if err != nil {
		xl.Errorf("remove those images with uid %v gid %v error:%v ", uid, id, err)
		return err
	}
	_, err = gColl.RemoveAll(bson.M{"uid": uid, "id": id})
	if err != nil {
		xl.Errorf("remove group with uid %v gid %v error:%v ", uid, id, err)
		return err
	}
	return nil
}

type imageGroupInMgo struct {
	*mgoutil.Collection
	hub FG.Hub
	uid uint32
	id  string
	hid FG.HubID
}

func (g imageGroupInMgo) Hub(context.Context) (FG.HubID, FG.Hub) { return g.hid, g.hub }

func (g imageGroupInMgo) Get(ctx context.Context, id string) (_ImageItem, error) {

	coll := g.CopySession()
	defer coll.CloseSession()

	var item _ImageItem
	err := coll.Find(bson.M{"uid": g.uid, "gid": g.id, "id": id}).One(&item)
	if mgo.ErrNotFound == err {
		return item, os.ErrNotExist
	}
	return item, err
}

func (g imageGroupInMgo) Add(ctx context.Context, items []_ImageItem, features [][]byte) []error {

	coll := g.CopySession()
	defer coll.CloseSession()

	_items := make([]interface{}, 0, len(items))
	for _, item := range items {
		// if len(item.ID) == 0 {
		// 	item.ID = xlog.GenReqId()
		// }
		_items = append(_items,
			struct {
				UID    uint32          `bson:"uid"`
				GID    string          `bson:"gid"`
				ID     string          `bson:"id"`
				Label  string          `bson:"label"`
				Etag   string          `bson:"etag"`
				URI    string          `bson:"uri"`
				Backup string          `bson:"backup"`
				Desc   json.RawMessage `bson:"desc,omitempty"`
			}{
				UID:    g.uid,
				GID:    g.id,
				ID:     item.ID,
				Label:  item.Label,
				Etag:   item.Etag,
				URI:    item.URI,
				Backup: item.Backup,
				Desc:   item.Desc,
			},
		)
	}

	errs := make([]error, len(_items))
	for i := range _items {
		errs[i] = coll.Insert(_items[i])
		if mgo.IsDup(errs[i]) {
			errs[i] = httputil.NewError(http.StatusBadRequest, `id already exists`)
		}
		if errs[i] == nil {
			errs[i] = g.hub.Set(ctx, g.hid, FG.FeatureID(items[i].ID), features[i])
		}
	}
	return errs
}

func (g imageGroupInMgo) Del(ctx context.Context, ids []string) error {

	coll := g.CopySession()
	defer coll.CloseSession()

	_, err := coll.RemoveAll(
		bson.M{"uid": g.uid, "gid": g.id, "id": bson.M{"$in": ids}},
	)
	return err
}

func (g imageGroupInMgo) All(ctx context.Context) ([]_ImageItem, error) {

	coll := g.CopySession()
	defer coll.CloseSession()

	items := make([]_ImageItem, 0)

	err := coll.
		Find(bson.M{"uid": g.uid, "gid": g.id}).
		Select(bson.M{"_id": 0, "uid": 0, "gid": 0}).
		All(&items)
	return items, err
}

func (g imageGroupInMgo) Count(ctx context.Context) (int, error) {

	coll := g.CopySession()
	defer coll.CloseSession()

	return coll.Find(bson.M{"uid": g.uid, "gid": g.id}).Count()
}

func (g imageGroupInMgo) Iter(ctx context.Context) (ImageGroupIter, error) {
	f := g.CopySession()
	//defer f.CloseSession()

	iter := f.
		Find(bson.M{"uid": g.uid, "gid": g.id}).
		Select(bson.M{"_id": 0, "uid": 0, "gid": 0}).
		Iter()

	return _ImageGroupIter{
		coll: f,
		iter: iter,
	}, nil
}

type _ImageGroupIter struct {
	iter *mgo.Iter
	coll mgoutil.Collection
}

func (fi _ImageGroupIter) Next(ctx context.Context) (*_ImageItem, bool) {
	var item _ImageItem
	ok := fi.iter.Next(&item)
	if ok {
		return &item, true
	}
	return nil, false
}

func (fi _ImageGroupIter) Error() error { return fi.iter.Err() }

func (fi _ImageGroupIter) Close() { fi.coll.CloseSession() }
