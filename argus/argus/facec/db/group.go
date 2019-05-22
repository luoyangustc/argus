package db

import (
	"context"
	"fmt"
	"time"

	"gopkg.in/mgo.v2"
	"gopkg.in/mgo.v2/bson"

	"github.com/qiniu/db/mgoutil.v3"
	"github.com/qiniu/xlog.v1"

	"qiniu.com/argus/argus/facec/dbbase"
)

// GroupInfo mapping collection groups
type GroupInfo struct {
	ID        bson.ObjectId `bson:"_id,omitempty"`
	Euid      string        `bson:"euid"`
	UID       string        `bson:"uid"`
	Group     Group         `bson:"group"`
	Modelv    string        `bson:"modelv"`
	CreatedAt time.Time     `bson:"created_at"`
}

// Group group detail
type Group struct {
	FaceCount int64    `bson:"face_count"`
	ID        int64    `bson:"final_id"`
	Refs      []Ref    `bson:"refs"`
	Faces     []FaceG  `bson:"faces"`
	Version   string   `bson:"version"`
	Files     []string `bson:",omitempty"` // TODO DELETE ME!!!
}

// Ref the cover of group
type Ref struct {
	File  string    `bson:"file"`
	Pts   [][]int64 `bson:"pts"`
	Score float64   `bson:"score"`
}

// FaceG face information in the group
type FaceG struct {
	ID                bson.ObjectId `bson:"face_id,omitempty"`
	File              string        `bson:"file"`
	Pts               [][]int64     `bson:"pts"`
	Score             float64       `bson:"score"`
	ClusterCenterDist float64       `bson:"cluster_center_dist"`
}

//const groupCollection = "groups"

// GroupDao groups collection dao
type GroupDao interface {
	Insert(ctx context.Context, groups []GroupInfo) error
	FindOne(ctx context.Context, uid, euid, version string, groupID int64) (GroupInfo, error)
	FindByVersion(ctx context.Context, uid, euid, version string) ([]GroupInfo, error)
	RemoveByVersion(ctx context.Context, uid, euid, version string) error

	// Deprecated
	AppendImages(uid, euid string, groupID int64, images []string, faceCount int) error
	Remove(uid, euid string, groupID int64) error
	MoveImage(uid, euid string, fromGroupID, toGroupID int64, images []string, facesNum map[string]int64) (bool, error)
	RemoveAll(uid, euid string) error
	RemoveAllByIDs(ids ...bson.ObjectId) error
	FindGroup(uid, euid string, groupID int64) (*GroupInfo, error)
	FindGroupByGroupID(uid, euid string, version string, groupID ...int64) ([]GroupInfo, error)
	FindGroups(uid, euid string, skip, limit int) ([]GroupInfo, error)
	FindIDs(uid, euid string) ([]bson.ObjectId, error)
	Exists(uid, euid string, groupIDs ...int64) ([]int64, error)
	RemoveGroups(uid, euid string, groupIDs ...int64) error
	RemoveGroupsWithVersion(uid, euid string, version string, groupIDs ...int64) error
	GetImages(uid, euid string, groupID ...int64) ([]string, int, error)
	SetRefs(uid, euid string, groupID int64, r Ref) error
	FindGroupWithExclude(uid, euid string, skip, limit int, excludeGroupIDs ...int) ([]GroupInfo, error)
	FindSimpleGroups(uid, euid string, skip, limit int) ([]GroupInfo, error)
	UpdateVersion(uid, euid, version string, exceptGroups ...int64) error
}

// NewGroupDao new group dao
func NewGroupDao() (GroupDao, error) {
	return &groupDao{coll: &collections.Groups}, nil
}

type groupDao struct{ coll *mgoutil.Collection }

func (d *groupDao) EnsureIndexes() error {
	coll := d.coll.CopySession()
	defer coll.CloseSession()

	if err := coll.EnsureIndex(
		mgo.Index{Key: []string{"uid", "euid", "group.version", "group.final_id"}, Unique: true},
	); err != nil {
		return err
	}
	return nil
}

func (d *groupDao) Insert(ctx context.Context, groups []GroupInfo) error {
	xl := xlog.FromContextSafe(ctx)

	groupLen := len(groups)
	if groupLen <= 0 {
		return nil
	}

	for i := 0; i < groupLen; i++ {
		if groups[i].ID == "" {
			groups[i].ID = bson.NewObjectId()
		}
	}

	c := d.coll.CopySession()
	defer c.CloseSession()

	insertParams := make([]interface{}, 0, len(groups))
	for i := 0; i < groupLen; i++ {
		insertParams = append(insertParams, &groups[i])
	}
	err := c.Insert(insertParams...)
	if err != nil {
		xl.Error("insert groups error:", err)
	}
	return err
}

func (d *groupDao) FindOne(ctx context.Context, uid, euid, version string, groupID int64) (GroupInfo, error) {
	coll := d.coll.CopySession()
	defer coll.CloseSession()

	var group GroupInfo
	err := coll.Find(
		bson.M{"uid": uid, "euid": euid, "group.version": version, "group.final_id": groupID},
	).One(&group)
	return group, err
}

func (d *groupDao) FindByVersion(ctx context.Context, uid, euid, version string) ([]GroupInfo, error) {
	coll := d.coll.CopySession()
	defer coll.CloseSession()

	var groups []GroupInfo
	err := coll.Find(bson.M{"uid": uid, "euid": euid, "group.version": version}).All(&groups)
	return groups, err
}

func (d *groupDao) RemoveByVersion(ctx context.Context, uid, euid, version string) error {
	coll := d.coll.CopySession()
	defer coll.CloseSession()

	xl := xlog.FromContextSafe(ctx)

	_, err := coll.RemoveAll(bson.M{"uid": uid, "euid": euid, "group.version": version})
	if err != nil {
		xl.Error("delete groups error:", err)
	}
	return err
}

////////////////////////////////////////////////////////////////////////////////
// Deprecated
func (d *groupDao) AppendImages(uid, euid string, groupID int64, images []string, faceCount int) error {
	c := d.coll.CopySession()
	defer c.CloseSession()

	err := c.Update(bson.M{
		"uid":            uid,
		"euid":           euid,
		"group.final_id": groupID,
	}, bson.M{
		"$push": bson.M{
			"group.files": bson.M{
				"$each": images,
			}},
		"$inc": bson.M{
			"group.face_count": faceCount,
		},
	})

	if err != nil {
		xlog.Errorf("", "append images to group error:", err)
	}
	return err
}

func (d *groupDao) Remove(uid, euid string, groupID int64) error {
	c := d.coll.CopySession()
	defer c.CloseSession()

	err := c.Remove(bson.M{"uid": uid, "euid": euid, "group.final_id": groupID})
	if err != nil {
		xlog.Errorf("", "remove group error:", err)
	}
	return err
}

func (d *groupDao) MoveImage(uid, euid string, fromGroupID, toGroupID int64, images []string, facesNum map[string]int64) (bool, error) {
	if len(images) == 0 {
		return false, nil
	}

	c := d.coll.CopySession()
	defer c.CloseSession()

	var srcImgs []struct {
		ID      bson.ObjectId `bson:"_id"`
		GrpInfo Group         `bson:"group"`
	}

	err := c.Find(bson.M{
		"uid":            uid,
		"euid":           euid,
		"group.final_id": bson.M{"$in": []int64{fromGroupID, toGroupID}},
	}).Select(bson.M{"_id": 1, "group.final_id": 1, "group.files": 1, "group.face_count": 1, "group.refs": 1}).All(&srcImgs)

	if err != nil {
		xlog.Errorf("", "query group's images error:", err)
		return false, err
	}

	if len(srcImgs) != 2 {
		return false, fmt.Errorf("some group not exist, from group:%d,to group:%d",
			fromGroupID, toGroupID)
	}

	var fromImages, toImages []string
	var fromID, toID bson.ObjectId
	var fromFaceCount, toFaceCount int64
	var fromRefs Ref
	var changeFromRefs bool
	if fromGroupID == srcImgs[0].GrpInfo.ID {
		fromID = srcImgs[0].ID
		toID = srcImgs[1].ID
		fromFaceCount = srcImgs[0].GrpInfo.FaceCount
		toFaceCount = srcImgs[1].GrpInfo.FaceCount
		fromRefs = srcImgs[0].GrpInfo.Refs[0]
	} else {
		fromID = srcImgs[1].ID
		toID = srcImgs[0].ID
		fromFaceCount = srcImgs[1].GrpInfo.FaceCount
		toFaceCount = srcImgs[0].GrpInfo.FaceCount
		fromRefs = srcImgs[1].GrpInfo.Refs[0]
	}

	var fromLeftImgs []string
	fromLeftIdex := make(map[string]int)
	for _, img := range images {
		fromLeftIdex[img] = 1
	}
	for _, img := range images {
		for _, fromImg := range fromImages {
			if img == fromImg {
				toImages = append(toImages, fromImg)
				toFaceCount += facesNum[img]
				fromFaceCount -= facesNum[img]
				if img == fromRefs.File {
					changeFromRefs = true
				}
				continue
			}
			if fromLeftIdex[fromImg] != 1 {
				fromLeftImgs = append(fromLeftImgs, fromImg)
				fromLeftIdex[fromImg] = 1
			}
		}
	}

	if len(fromLeftImgs) == len(fromImages) {
		return false, fmt.Errorf("no images:%v, in group:%d", images, fromGroupID)
	}
	if fromFaceCount < 0 {
		return false, fmt.Errorf("unexpected error occure,the group.face_count becomes negative:%d", fromFaceCount)
	}

	if len(fromLeftImgs) == 0 { //remove empty group
		err = d.RemoveGroups(uid, euid, fromGroupID)
		changeFromRefs = false
		if err != nil {
			xlog.Errorf("", "remove `from group` error:%v", err)
			return false, err
		}
	} else {
		err = c.Update(
			bson.M{"_id": fromID},
			bson.M{"$set": bson.M{"group.files": fromLeftImgs, "group.face_count": fromFaceCount, "created_at": time.Now()}})
		if err != nil {
			xlog.Errorf("", "update `from group` image error:%v", err)
			return false, err
		}
	}

	err = c.Update(bson.M{"_id": toID}, bson.M{"$set": bson.M{"group.files": toImages, "group.face_count": toFaceCount, "created_at": time.Now()}})
	if err != nil {
		xlog.Errorf("", "update `to group` image error:%v", err)
		return false, err
	}

	return changeFromRefs, nil
}

func (d *groupDao) RemoveAll(uid, euid string) error {
	c := d.coll.CopySession()
	defer c.CloseSession()

	_, err := c.RemoveAll(bson.M{"uid": uid, "euid": euid})
	if err != nil {
		xlog.Errorf("", "delete groups error:", err)
	}
	return err
}

func (d *groupDao) FindGroup(uid, euid string, groupID int64) (*GroupInfo, error) {
	c := d.coll.CopySession()
	defer c.CloseSession()

	var group GroupInfo
	err := c.Find(bson.M{
		"uid":            uid,
		"euid":           euid,
		"group.final_id": groupID,
	}).One(&group)
	if err != nil {
		xlog.Errorf("", "find group error:%v", err)
		return nil, err
	}

	return &group, nil
}

func (d *groupDao) FindGroupByGroupID(uid, euid string, version string, groupID ...int64) ([]GroupInfo, error) {
	c := d.coll.CopySession()
	defer c.CloseSession()

	var groups []GroupInfo
	err := c.Find(bson.M{
		"uid":            uid,
		"euid":           euid,
		"group.final_id": bson.M{"$in": groupID},
		"group.version":  version,
	}).All(&groups)

	if err != nil {
		xlog.Errorf("", "find groupS error:%v", err)
		return nil, err
	}

	return groups, nil
}

func (d *groupDao) FindGroups(uid, euid string, skip, limit int) ([]GroupInfo, error) {
	c := d.coll.CopySession()
	defer c.CloseSession()

	if limit > dbbase.MaxLimit {
		limit = dbbase.MaxLimit
	}

	var groups []GroupInfo
	err := c.Find(bson.M{
		"uid":  uid,
		"euid": euid,
	}).Skip(skip).Limit(limit).All(&groups)

	if err != nil {
		xlog.Errorf("", "find groupS error:%v", err)
		return nil, err
	}

	return groups, nil
}

func (d *groupDao) RemoveAllByIDs(ids ...bson.ObjectId) error {
	if len(ids) == 0 {
		return nil
	}

	c := d.coll.CopySession()
	defer c.CloseSession()

	_, err := c.RemoveAll(bson.M{"_id": bson.M{"$in": ids}})
	if err != nil {
		xlog.Errorf("", "delete groups error:", err)
	}

	return err
}

func (d *groupDao) FindIDs(uid, euid string) ([]bson.ObjectId, error) {
	c := d.coll.CopySession()
	defer c.CloseSession()

	var ids []struct {
		ID bson.ObjectId `bson:"_id"`
	}
	err := c.Find(bson.M{
		"uid":  uid,
		"euid": euid,
	}).Select(bson.M{"_id": 1}).All(&ids)

	if err != nil {
		xlog.Errorf("", "find ids error:%v", err)
		return nil, err
	}

	var ret []bson.ObjectId
	for _, id := range ids {
		ret = append(ret, id.ID)
	}

	return ret, nil
}

func (d *groupDao) Exists(uid, euid string, groupIDs ...int64) ([]int64, error) {
	c := d.coll.CopySession()
	defer c.CloseSession()

	var ids []bson.M
	err := c.Find(bson.M{
		"uid":  uid,
		"euid": euid,
		"group.final_id": bson.M{
			"$in": groupIDs,
		},
	}).Select(bson.M{"group.final_id": 1, "_id": 0}).All(&ids)

	if err != nil || len(ids) == 0 {
		xlog.Errorf("", "find ids error:%v", err)
		return nil, err
	}

	var gids []int64
	for _, d := range ids {
		gids = append(gids, int64(d["group"].(bson.M)["final_id"].(float64)))
	}

	return gids, nil
}

func (d *groupDao) RemoveGroups(uid, euid string, groupIDs ...int64) error {
	c := d.coll.CopySession()
	defer c.CloseSession()

	_, err := c.RemoveAll(bson.M{
		"uid":  uid,
		"euid": euid,
		"group.final_id": bson.M{
			"$in": groupIDs,
		},
	})
	if err != nil {
		xlog.Errorf("", "find ids error:%v", err)
		return err
	}
	return nil
}

func (d *groupDao) RemoveGroupsWithVersion(uid, euid string, version string, groupIDs ...int64) error {
	c := d.coll.CopySession()
	defer c.CloseSession()

	_, err := c.RemoveAll(bson.M{
		"uid":           uid,
		"euid":          euid,
		"group.version": version,
		"group.final_id": bson.M{
			"$in": groupIDs,
		},
	})
	if err != nil {
		xlog.Errorf("", "find ids error:%v", err)
		return err
	}
	return nil
}

//return images, face nums of the images
func (d *groupDao) GetImages(uid, euid string, groupID ...int64) ([]string, int, error) {
	c := d.coll.CopySession()
	defer c.CloseSession()

	var statistic []struct {
		Facex  int      `bson:"_id"`
		Images []string `bson:"images"`
	}

	err := c.Pipe([]bson.M{
		bson.M{"$match": bson.M{
			"uid":            uid,
			"euid":           euid,
			"group.final_id": bson.M{"$in": groupID},
		}},
		bson.M{"$group": bson.M{
			"_id":   "$uid_$euid",
			"image": bson.M{"$push": "$group.files"},
			"faces": bson.M{"$sum": "$group.face_count"},
		}},
		bson.M{"$unwind": "$image"},
		bson.M{"$unwind": "$image"},
		bson.M{"$project": bson.M{
			"_id":   0,
			"faces": "$faces",
			"image": "$image",
		}},
		bson.M{"$group": bson.M{
			"_id":    "$faces",
			"images": bson.M{"$addToSet": "$image"},
		}},
	}).All(&statistic)

	if err != nil || len(statistic) == 0 {
		xlog.Errorf("", "find ids error:%v", err)
		return nil, 0, err
	}

	return statistic[0].Images, statistic[0].Facex, nil
}

func (d *groupDao) SetRefs(uid, euid string, groupID int64, r Ref) error {
	c := d.coll.CopySession()
	defer c.CloseSession()

	err := c.Update(bson.M{"uid": uid, "euid": euid, "group.final_id": groupID}, bson.M{"$set": bson.M{"group.refs": r}})
	if err != nil {
		xlog.Errorf("", "update refs:%v of group:%v, uid:%v,euid:%v error", r, groupID, uid, euid)
		return err
	}
	return nil
}

func (d *groupDao) FindGroupWithExclude(uid, euid string,
	skip, limit int,
	excludeGroupIDs ...int) ([]GroupInfo, error) {
	c := d.coll.CopySession()
	defer c.CloseSession()

	if limit > dbbase.MaxLimit {
		limit = dbbase.MaxLimit
	}

	cond := bson.M{
		"uid":  uid,
		"euid": euid,
	}
	if len(excludeGroupIDs) > 0 {
		cond["group.final_id"] = bson.M{"$nin": excludeGroupIDs}
	}

	var groups []GroupInfo
	err := c.Find(cond).Skip(skip).Limit(limit).All(&groups)

	if err != nil {
		xlog.Errorf("", "find groupS with exclude ids error:%v", err)
		return nil, err
	}

	return groups, nil
}

// FindSimpleGroups return group information without faces detail
func (d *groupDao) FindSimpleGroups(uid, euid string, skip, limit int) ([]GroupInfo, error) {
	c := d.coll.CopySession()
	defer c.CloseSession()

	if limit > dbbase.MaxLimit {
		limit = dbbase.MaxLimit
	}

	var groups []GroupInfo
	err := c.Find(bson.M{
		"uid":  uid,
		"euid": euid,
	}).Select(bson.M{"group.faces": 0}).Skip(skip).Limit(limit).All(&groups)

	if err != nil {
		xlog.Errorf("", "find groupS error:%v", err)
		return nil, err
	}

	return groups, nil
}

func (d *groupDao) UpdateVersion(uid, euid, version string, exceptGroups ...int64) error {
	c := d.coll.CopySession()
	defer c.CloseSession()
	err := c.Update(bson.M{"uid": uid, "euid": euid, "group.final_id": bson.M{"$nin": exceptGroups}}, bson.M{"$set": bson.M{"group.version": version}})
	if err != nil {
		xlog.Errorf("", "update verion:%v of all groups of uid:%v,euid:%v error", version, uid, euid)
		return err
	}
	return nil

}
