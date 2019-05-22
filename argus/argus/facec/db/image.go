package db

import (
	"time"

	"gopkg.in/mgo.v2"
	"gopkg.in/mgo.v2/bson"

	"github.com/qiniu/db/mgoutil.v3"
	xlog "github.com/qiniu/xlog.v1"
)

//const imageCollection = "images"

// Image mapping collection images
type Image struct {
	ID        bson.ObjectId   `bson:"_id,omitempty"`
	Euid      string          `bson:"euid"`
	Faces     []bson.ObjectId `bson:"faces"`
	File      string          `bson:"file"`
	UID       string          `bson:"uid"`
	CreatedAt time.Time       `bson:"created_at"`
}

// ImageDao images collection dao
type ImageDao interface {
	FindImages(uid, euid string) ([]Image, error)
	FindFaces(uid, euid string, files ...string) (map[string][]bson.ObjectId, error)
	FindImageByFiles(uid, euid string, files []string) ([]Image, error)
	Insert(image []Image) error
	Remove(uid, euid string) error
	Exists(uid, euid, file string) (ok bool, err error)
}

// NewImageDao new image dao
func NewImageDao() (ImageDao, error) {
	return &imageDao{coll: &collections.Images}, nil
}

type imageDao struct{ coll *mgoutil.Collection }

func (d *imageDao) EnsureIndexes() error {
	coll := d.coll.CopySession()
	defer coll.CloseSession()

	if err := coll.EnsureIndex(mgo.Index{Key: []string{"uid", "euid", "file"}, Unique: true}); err != nil {
		return err
	}
	return nil
}

func (d *imageDao) FindImages(uid, euid string) ([]Image, error) {
	c := d.coll.CopySession()
	defer c.CloseSession()

	var ret []Image
	err := c.Find(bson.M{"euid": euid, "uid": uid}).All(&ret)
	if err != nil {
		xlog.Errorf("", "query user's image error", err)
		return nil, err
	}

	return ret, nil
}

func (d *imageDao) FindFaces(uid, euid string,
	files ...string) (map[string][]bson.ObjectId, error) {
	c := d.coll.CopySession()
	defer c.CloseSession()

	var faces []struct {
		Faces []bson.ObjectId `bson:"faces"`
		File  string          `bson:"file"`
	}

	cond := bson.M{
		"uid":     uid,
		"euid":    euid,
		"faces.0": bson.M{"$exists": true},
	}
	if len(files) > 0 {
		cond["file"] = bson.M{"$in": files}
	}
	err := c.Find(cond).Select(bson.M{"faces": 1, "file": 1}).All(&faces)
	if err != nil {
		xlog.Errorf("", "query user's faces error", err)
		return nil, err
	}

	ret := make(map[string][]bson.ObjectId, len(faces))
	for _, f := range faces {
		ret[f.File] = f.Faces
	}

	return ret, nil
}

func (d *imageDao) Insert(images []Image) error {
	if len(images) <= 0 {
		return nil
	}
	c := d.coll.CopySession()
	defer c.CloseSession()
	len := len(images)
	for i := 0; i < len; i++ {
		if images[i].ID == "" {
			images[i].ID = bson.NewObjectId()
		}
	}

	var insertRet []interface{}
	for i := 0; i < len; i++ {
		insertRet = append(insertRet, &images[i])
	}
	err := c.Insert(insertRet...)
	if err != nil {
		xlog.Errorf("", "insert image error:%v", err)
	}

	return err
}

func (d *imageDao) Remove(uid, euid string) error {
	c := d.coll.CopySession()
	defer c.CloseSession()

	_, err := c.RemoveAll(bson.M{"uid": uid, "euid": euid})

	if err != nil {
		xlog.Errorf("", "remove images error:%v", err)
	}

	return err
}
func (d *imageDao) FindImageByFiles(uid, euid string, files []string) ([]Image, error) {
	c := d.coll.CopySession()
	defer c.CloseSession()

	var ret []Image
	err := c.Find(bson.M{"euid": euid, "uid": uid, "file": bson.M{"$in": files}}).All(&ret)
	if err != nil {
		xlog.Errorf("", "query user's image error", err)
		return nil, err
	}

	return ret, nil
}

func (d *imageDao) Exists(uid, euid, file string) (ok bool, err error) {
	c := d.coll.CopySession()
	defer c.CloseSession()

	n, err := c.Find(bson.M{"uid": uid, "euid": euid, "file": file}).Count()
	return n > 0, err
}
