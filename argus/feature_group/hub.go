package feature_group

import (
	"context"
	"reflect"

	"github.com/qiniu/xlog.v1"

	"github.com/pkg/errors"

	"gopkg.in/mgo.v2"
	"gopkg.in/mgo.v2/bson"

	"github.com/qiniu/db/mgoutil.v3"
)

type HubID string
type FeatureID string
type HubVersion int

type Hub interface {
	New(ctx context.Context, chunkSize uint64, featureVersion FeatureVersion) (HubID, error)
	Remove(ctx context.Context, hid HubID) error

	Set(ctx context.Context, hid HubID, fid FeatureID, p []byte) error
	Del(ctx context.Context, hid HubID, fid FeatureID) error

	Find(ctx context.Context, hid HubID, ver HubVersion, index int) (FeatureID, error)
	All(ctx context.Context, hid HubID, blockSize int) ([]FeatureBlock, error)
	Fetch(ctx context.Context, hid HubID, ver HubVersion, from, to int) (FeatureBlockIter, error)

	FeatureVersion(ctx context.Context, hid HubID) (FeatureVersion, error)

	Clean() // FOR UT
}

type FeatureBlock struct {
	Hid       HubID
	Ver       HubVersion
	From, To  int
	ChunkSize uint64
}

type FeatureBlockIter interface {
	Next(ctx context.Context) ([]byte, bool)
	Error() error
	Close()
}

//----------------------------------------------------------------------------//

type _HubItem struct {
	ID            bson.ObjectId  `bson:"_id,omitempty"`
	ChunkSize     uint64         `bson:"chunk_size"`
	FeatureVersin FeatureVersion `bson:"feature_version"`
	Cursor        int            `bson:"cursor"`
	Version       HubVersion     `bson:"version"`
}

type _FeatureItem struct {
	Hid     HubID      `bson:"hid"`
	Fid     FeatureID  `bson:"fid"`
	Index   int        `bson:"index"`
	Version HubVersion `bson:"version"`
	Feature []byte     `bson:"feature"`
}

var _ Hub = &_HubInMgo{}

type _HubInMgo struct {
	hubs     *mgoutil.Collection
	features *mgoutil.Collection
}

func NewHubInMgo(mgoConf *mgoutil.Config, colls interface{}) (*_HubInMgo, error) {

	sess, err := mgoutil.Open(colls, mgoConf)
	if err != nil {
		return nil, err
	}
	// sess.Close()
	sess.SetPoolLimit(100) // DefaultCollSessionPoolLimit
	var (
		hubs     = reflect.ValueOf(colls).Elem().FieldByName("Hubs").Interface().(mgoutil.Collection)
		features = reflect.ValueOf(colls).Elem().FieldByName("Features").Interface().(mgoutil.Collection)
	)
	err = features.EnsureIndex(mgo.Index{Key: []string{"hid", "fid"}, Unique: true})
	if err != nil {
		return nil, errors.Wrap(err, "Groups.EnsureIndex hid fid")
	}
	err = features.EnsureIndex(mgo.Index{Key: []string{"hid", "index", "version"}, Unique: true})
	if err != nil {
		return nil, errors.Wrap(err, "Groups.EnsureIndex hid index version")
	}
	err = features.EnsureIndex(mgo.Index{Key: []string{"hid", "index", "-version"}})
	if err != nil {
		return nil, errors.Wrap(err, "Groups.EnsureIndex hid index -version")
	}

	return &_HubInMgo{hubs: &hubs, features: &features}, nil
}

func (h _HubInMgo) New(ctx context.Context, chunkSize uint64, featureVersion FeatureVersion) (HubID, error) {
	hid := bson.NewObjectId()

	coll := h.hubs.CopySession()
	defer coll.CloseSession()

	err := coll.Insert(_HubItem{ID: hid, ChunkSize: chunkSize, FeatureVersin: featureVersion})
	return HubID(hid.Hex()), err
}

func (h _HubInMgo) Remove(ctx context.Context, hid HubID) error {
	coll := h.features.CopySession()
	defer coll.CloseSession()
	_, err := coll.RemoveAll(bson.M{"hid": hid})
	if err != nil {
		return errors.Wrap(err, "coll.RemoveAll")
	}
	coll2 := h.hubs.CopySession()
	defer coll2.CloseSession()
	err = coll2.RemoveId(bson.ObjectIdHex(string(hid)))

	return err
}

func (h _HubInMgo) Set(ctx context.Context, hid HubID, fid FeatureID, p []byte) error {
	hColl := h.hubs.CopySession()
	defer hColl.CloseSession()

	var item _HubItem
	_, err := hColl.FindId(bson.ObjectIdHex(string(hid))).
		Apply(
			mgo.Change{Update: bson.M{"$inc": bson.M{"cursor": 1, "version": 1}}},
			&item,
		)
	if err != nil {
		return errors.Wrapf(err, "hColl.FindId %v", hid)
	}
	fColl := h.features.CopySession()
	defer fColl.CloseSession()

	_, err = fColl.Upsert(
		bson.M{"hid": hid, "fid": fid},
		_FeatureItem{Hid: hid, Fid: fid, Index: item.Cursor, Version: item.Version, Feature: p},
	)

	return err
}

func (h _HubInMgo) Del(ctx context.Context, hid HubID, fid FeatureID) error {
	coll := h.features.CopySession()
	defer coll.CloseSession()

	err := coll.Remove(bson.M{"hid": hid, "fid": fid})
	if err != nil {
		return errors.Wrap(err, "coll.Remove")
	}
	return nil
}

func (h _HubInMgo) Find(ctx context.Context, hid HubID, ver HubVersion, index int) (FeatureID, error) {
	coll := h.features.CopySession()
	defer coll.CloseSession()

	var ret = struct {
		Fid FeatureID `bson:"fid"`
	}{}
	err := coll.
		Find(bson.M{"hid": hid, "version": bson.M{"$lte": ver}, "index": index}).
		Sort("-version").
		One(&ret)
	if err != nil {
		return "", err
	}
	return ret.Fid, nil
}

func (h _HubInMgo) FeatureVersion(ctx context.Context, hid HubID) (FeatureVersion, error) {
	coll := h.hubs.CopySession()
	defer coll.CloseSession()

	var hub _HubItem
	if err := coll.FindId(bson.ObjectIdHex(string(hid))).One(&hub); err != nil {
		return "", err
	}
	return hub.FeatureVersin, nil
}

func (h _HubInMgo) All(ctx context.Context, hid HubID, blockSize int) ([]FeatureBlock, error) {
	coll := h.hubs.CopySession()
	defer coll.CloseSession()

	coll2 := h.features.CopySession()
	defer coll2.CloseSession()

	var hub _HubItem
	err := coll.FindId(bson.ObjectIdHex(string(hid))).One(&hub)
	if err != nil {
		return nil, errors.Wrapf(err, "hColl.FindId %v", hid)
	}

	var ret = make([]FeatureBlock, 0, hub.Cursor/blockSize+1)
	for index := 0; index < hub.Cursor; index += blockSize {
		to := index + blockSize
		if to >= hub.Cursor {
			to = hub.Cursor
		}
		var item _FeatureItem
		for i := to - 1; i >= index; i-- {
			err := coll2.Find(bson.M{"hid": hid, "index": i}).One(&item)
			if err == nil {
				break
			}
			if err == mgo.ErrNotFound {
				continue
			}
			return nil, err
		}
		ret = append(ret,
			FeatureBlock{
				Hid: hid, Ver: item.Version,
				From: index, To: to,
				ChunkSize: hub.ChunkSize,
			})
	}

	return ret, nil
}

type _FeatureBlockIterInMgo struct {
	coll             mgoutil.Collection
	item             *_FeatureItem
	from, to, cursor int
	iter             *mgo.Iter
	end              bool
	emptyFeature     []byte
}

func (fi *_FeatureBlockIterInMgo) Next(ctx context.Context) ([]byte, bool) {
	if fi.cursor >= fi.to {
		return nil, false
	}
	if fi.cursor < fi.from {
		fi.cursor = fi.from
	}
	p := fi.emptyFeature
	if fi.item != nil {
		if fi.item.Index == fi.cursor {
			p = fi.item.Feature
			fi.item = nil
		}
		fi.cursor++
		return p, true
	}
	for !fi.end {
		var item _FeatureItem
		ok := fi.iter.Next(&item)
		if !ok {
			fi.end = true
			break
		}
		if item.Index < fi.cursor {
			continue
		}
		if item.Index > fi.cursor {
			fi.item = &item
		} else {
			p = item.Feature
		}
		break
	}
	fi.cursor++
	return p, true
}

func (fi _FeatureBlockIterInMgo) Error() error { return fi.iter.Err() }
func (fi _FeatureBlockIterInMgo) Close()       { fi.coll.CloseSession() }

func (h _HubInMgo) Fetch(
	ctx context.Context, hid HubID, ver HubVersion, from, to int,
) (FeatureBlockIter, error) {
	hColl := h.hubs.CopySession()
	defer hColl.CloseSession()

	var item _HubItem
	err := hColl.FindId(bson.ObjectIdHex(string(hid))).One(&item)
	if err != nil {
		return nil, errors.Wrapf(err, "hColl.FindId %v", hid)
	}

	coll := h.features.CopySession()
	// defer coll.CloseSession()

	iter := coll.
		Find(bson.M{"hid": hid, "version": bson.M{"$lte": ver}, "index": bson.M{"$gte": from, "$lt": to}}).
		Sort("index", "-version").
		Iter()

	return &_FeatureBlockIterInMgo{
		coll: coll,
		from: from, to: to, cursor: from - 1,
		iter:         iter,
		emptyFeature: emptyFeature(item.ChunkSize),
	}, nil
}

func emptyFeature(chunkSize uint64) []byte { return make([]byte, chunkSize) }

func (h _HubInMgo) Clean() {
	xl := xlog.NewWith("main")
	_, err := h.hubs.RemoveAll(nil)
	if err != nil {
		xl.Error(err)
	}
	_, err = h.features.RemoveAll(nil)
	if err != nil {
		xl.Error(err)
	}
}
