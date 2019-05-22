package hub

import (
	"time"

	"qiniu.com/argus/tuso/proto"

	"github.com/qiniu/db/mgoutil.v3"
	"gopkg.in/mgo.v2/bson"
)

type db struct {
	Hub      mgoutil.Collection `coll:"hub"`
	OpLog    mgoutil.Collection `coll:"oplog"`
	FileMeta mgoutil.Collection `coll:"file_meta"`
	HubMeta  mgoutil.Collection `coll:"hub_meta"`
}

func (d *db) createIndex() {
	d.Hub.EnsureIndexes("hub_name :unique")
	d.OpLog.EnsureIndexes("create_time", "status,op")
	// TODO: hub_name,index,offset 是否需要unique?
	d.FileMeta.EnsureIndexes("hub_name,key:unique", "hub_name,index,offset")
	d.HubMeta.EnsureIndexes("hub_name :unique")
}

type dHub struct {
	HubName string `bson:"hub_name"`
	UID     uint32 `bson:"uid"`
	Bucket  string `bson:"bucket"`
	Prefix  string `bson:"prefix"`
}

type OpStatus string

var OptatusInit = OpStatus("INIT")
var OptatusEvaling = OpStatus("EVALING")
var OptatusEvaled = OpStatus("EVALED")
var OptatusEVALERROR = OpStatus("EVALERROR")

var OptatusEnum = []OpStatus{OptatusInit, OptatusEvaling, OptatusEvaled}

func OpStatusFromString(s string) (OpStatus, error) {
	r := OpStatus(s)
	for _, v := range OptatusEnum {
		if v == r {
			return v, nil
		}
	}
	return OpStatus(""), ErrBadStatus
}

type OpKind string

var OpKindAdd = OpKind("ADD")
var OpKindUpdate = OpKind("UPDATE")
var OpKindDelete = OpKind("DELETE")

var OpKindEnum = []OpKind{OpKindAdd, OpKindUpdate, OpKindDelete}

func OpKindEnumFromString(s string) (OpKind, error) {
	r := OpKind(s)
	for _, v := range OpKindEnum {
		if v == r {
			return v, nil
		}
	}
	return OpKind(""), ErrBadOp
}

type FileMetaStatus string

var FileMetaStatusInit = FileMetaStatus("INIT")
var FileMetaStatusOK = FileMetaStatus("OK")
var FileMetaStatusDeleted = FileMetaStatus("DELETED")

type dOpLog struct {
	ID         bson.ObjectId `bson:"_id,omitempty"`
	Op         OpKind        `bson:"op"`
	Status     OpStatus      `bson:"status"`
	HubName    string        `bson:"hub_name"`
	Key        string        `bson:"key"`
	CreateTime time.Time     `bson:"create_time"`
	Md5        string        `bson:"md5,omitempty"`
	Feature    []byte        `bson:"feature,omitempty"`
}

func (d db) createHub(hub dHub) error {
	coll := d.Hub.CopySession()
	defer coll.CloseSession()
	return coll.Insert(hub)
}

func (d db) createHubMeta(hub dHubMeta) error {
	coll := d.HubMeta.CopySession()
	defer coll.CloseSession()
	return coll.Insert(hub)
}

func (d db) insertOpLog(o dOpLog) error {
	coll := d.OpLog.CopySession()
	defer coll.CloseSession()
	return coll.Insert(o)
}

func (d db) isHubOwner(uid uint32, hub string) error {
	coll := d.Hub.CopySession()
	defer coll.CloseSession()
	n, err := coll.Find(bson.M{"uid": uid, "hub_name": hub}).Count()
	if err != nil {
		return err
	}
	if n == 0 {
		return ErrHubNotFound
	}
	return nil
}

type dOpLogPipeResult struct {
	HubName string `bson:"_id"`
	Cnt     int    `bson:"cnt"`
}

func (d db) countOpLogShouldUploadUser() ([]*dOpLogPipeResult, error) {
	coll := d.OpLog.CopySession()
	defer coll.CloseSession()
	/*
		db.getCollection("oplog").aggregate([
		  {
		    $project: {
		      status: 1,
		      op: 1,
		      hub_name: 1
		    }
		  },
		  {
		    $match: {
		      status: "EVALED",
		      op: "ADD"
		    }
		  },
		  {
		    $group: {
		      _id: "$hub_name",
		      cnt: { $sum: 1 }
		    }
		  },
		  {
		    $match: {
		      cnt: { $gte: 256 }
		    }
		  }
		]);
	*/
	var r []*dOpLogPipeResult
	// TODO: 效率问题？ limit？
	err := coll.Pipe([]bson.M{
		{"$project": bson.M{"status": 1, "op": 1, "hub_name": 1}},
		{"$match": bson.M{"status": OptatusEvaled, "op": OpKindAdd}},
		{"$group": bson.M{"_id": "$hub_name", "cnt": bson.M{"$sum": 1}}},
		{"$match": bson.M{"cnt": bson.M{"$gte": proto.KodoBlockFeatureSize}}},
	}).All(&r)
	if err != nil {
		return nil, err
	}
	return r, nil
}

func (d db) findOneBlockOpLogShouldUpload(hubName string) ([]dOpLog, error) {
	coll := d.OpLog.CopySession()
	defer coll.CloseSession()
	var oplog []dOpLog
	err := coll.Find(bson.M{"status": OptatusEvaled, "op": OpKindAdd, "hub_name": hubName}).Limit(proto.KodoBlockFeatureSize).All(&oplog)
	return oplog, err
}

func (d db) removeMultiOpLog(opLogIds []bson.ObjectId) (removed int, err error) {
	coll := d.OpLog.CopySession()
	defer coll.CloseSession()
	info, err := coll.RemoveAll(bson.M{"_id": bson.M{"$in": opLogIds}})
	return info.Removed, err
}

func (d db) updateOplogStatus(id bson.ObjectId, status OpStatus) error {
	coll := d.OpLog.CopySession()
	defer coll.CloseSession()
	return coll.UpdateId(id, bson.M{"$set": bson.M{"status": status}})
}

func (d db) updateOplogStatusAndFeature(id bson.ObjectId, status OpStatus, feature []byte, md5 string) error {
	coll := d.OpLog.CopySession()
	defer coll.CloseSession()
	return coll.UpdateId(id, bson.M{"$set": bson.M{
		"status":  status,
		"feature": feature,
		"md5":     md5,
	}})
}

type dFileMeta struct {
	ID                bson.ObjectId  `bson:"_id,omitempty"`
	HubName           string         `bson:"hub_name"`
	Key               string         `bson:"key"`
	UpdateTime        time.Time      `bson:"update_time"`
	Status            FileMetaStatus `bson:"status"`
	Md5               string         `bson:"md5,omitempty"`
	FeatureFileIndex  int            `bson:"index"`
	FeatureFileOffset int            `bson:"offset"`
}

type dHubMeta struct {
	ID               bson.ObjectId `bson:"_id,omitempty"`
	HubName          string        `bson:"hub_name"`
	FeatureVersion   int           `bson:"version"`
	FeatureFileIndex int           `bson:"index"`
}

type dOpLogStatResult struct {
	FieldValue string `bson:"_id"`
	Cnt        int    `bson:"cnt"`
}

func (d db) opLogStat(limit int, field string) ([]*dOpLogStatResult, error) {
	coll := d.OpLog.CopySession()
	defer coll.CloseSession()
	var r []*dOpLogStatResult
	err := coll.Pipe([]bson.M{
		{"$limit": limit},
		{"$group": bson.M{"_id": "$" + field, "cnt": bson.M{"$sum": 1}}},
	}).All(&r)
	if err != nil {
		return nil, err
	}
	return r, nil
}

func (d db) findHubInfo(hub string) (dHub, error) {
	coll := d.Hub.CopySession()
	defer coll.CloseSession()
	var r dHub
	err := coll.Find(bson.M{"hub_name": hub}).One(&r)
	return r, err
}

func (d db) findHubMeta(hub string) (dHubMeta, error) {
	coll := d.HubMeta.CopySession()
	defer coll.CloseSession()
	var r dHubMeta
	err := coll.Find(bson.M{"hub_name": hub}).One(&r)
	return r, err
}

func (d db) listHubByUID(uid uint32) ([]dHub, error) {
	coll := d.Hub.CopySession()
	defer coll.CloseSession()
	var r []dHub
	err := coll.Find(bson.M{"uid": uid}).All(&r)
	return r, err
}

func (d db) countFileMetaByUid(hub string) (n int, err error) {
	coll := d.FileMeta.CopySession()
	defer coll.CloseSession()
	n, err = coll.Find(bson.M{"hub_name": hub}).Count()
	return n, err
}
