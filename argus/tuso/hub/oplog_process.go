package hub

import (
	"context"
	"encoding/json"
	"fmt"
	"sync"
	"time"

	"github.com/pkg/errors"

	"github.com/qiniu/xlog.v1"

	"gopkg.in/mgo.v2/bson"
	"qiniu.com/argus/tuso/proto"
)

var xl = xlog.NewWith("opLogProcess")

type opLogProcess struct {
	db             *db
	api            proto.ImageFeatureApi
	concurrencyNum int
	uploader       uploader
	l              sync.RWMutex
	batchSize      int
}

func (o *opLogProcess) getConcurrencyNum() int {
	o.l.RLock()
	n := o.concurrencyNum
	o.l.RUnlock()
	return n
}

func (o *opLogProcess) SetConcurrencyNum(n int) {
	o.l.Lock()
	o.concurrencyNum = n
	o.l.Unlock()
}

func (o *opLogProcess) processOpLog(ctx context.Context, opLog dOpLog) error {
	err := o.db.updateOplogStatus(opLog.ID, OptatusEvaling)
	if err != nil {
		return errors.Wrap(err, "db.OpLog.UpdateId OptatusEvaling")
	}
	// TODO: timeout
	// TODO: Batch
	hubInfo, err := o.db.findHubInfo(opLog.HubName)
	if err != nil {
		return errors.Wrap(err, "findHubInfo")
	}
	req := proto.PostEvalFeatureReq{
		Image: proto.Image{
			Key:    opLog.Key,
			Bucket: hubInfo.Bucket,
			Uid:    hubInfo.UID,
		},
	}
	resp, err := o.api.PostEvalFeature(ctx, req)
	if err != nil {
		xl.Warn("api.PostEvalFeature error", err, req)
		err2 := o.db.updateOplogStatus(opLog.ID, OptatusEVALERROR)
		if err2 != nil {
			return errors.Wrapf(err, "db.OpLog.UpdateId %v", err2)
		}
		return nil
	}
	err = o.db.updateOplogStatusAndFeature(opLog.ID, OptatusEvaled, resp.Feature, resp.Md5)
	if err != nil {
		return errors.Wrap(err, "db.OpLog.UpdateId done")
	}
	return nil
}

func (o *opLogProcess) processAllOpLog(ctx context.Context) (oplogProcessNum int, err error) {
	w := sync.WaitGroup{}
	// TODO: use tailable iterator?
	iter := o.db.OpLog.Find(bson.M{"status": OptatusInit, "op": OpKindAdd}).Iter()
	var opLog dOpLog
	ch := make(chan struct{}, o.getConcurrencyNum())
	defer iter.Close()
	cnt := 0
	for iter.Next(&opLog) {
		// 让并发数目修改不需要太久才能生效
		cnt++
		if cnt > o.batchSize {
			break
		}
		oplogProcessNum++
		select {
		case <-ctx.Done():
			return oplogProcessNum, ctx.Err()
		default:
		}
		ch <- struct{}{}
		w.Add(1)
		go func(opLog dOpLog) {
			defer func() {
				<-ch
				w.Done()
			}()
			err := o.processOpLog(ctx, opLog)
			if err != nil {
				xl.Error("o.processOpLog", err)
			}
		}(opLog)
	}
	if err := iter.Err(); err != nil {
		return oplogProcessNum, errors.Wrap(err, "processOpLog OpLog.Find")
	}
	w.Wait()
	return oplogProcessNum, nil
}

func (o *opLogProcess) processAllOpLogLoop(ctx context.Context) error {
	for {
		select {
		case <-ctx.Done():
			return ctx.Err()
		default:
		}
		start := time.Now()
		n, err := o.processAllOpLog(ctx)
		if err != nil {
			xl.Error("o.processAllOpLog", err)
		}
		since := time.Since(start)
		if n == 0 {
			xl.Debug("o.processAllOpLog over, use time", since, "record", n)
		} else {
			xl.Info("o.processAllOpLog over, use time", since, "record", n)
		}
		if n == 0 {
			// TODO:conf
			time.Sleep(time.Second * 10)
		}
	}
}

func (o *opLogProcess) uploadFeatureToKodo(ctx context.Context, hubName string, version int, index int, feature []byte) (err error) {
	// TODO:upload
	key := fmt.Sprintf("%v/%v/%v", hubName, version, index)
	err = o.uploader.upload(ctx, key, feature)
	if err != nil {
		return err
	}
	return nil
}

func (o *opLogProcess) uploadKodoOneHub(ctx context.Context, hubName string) (successedNum int, index int, err error) {
	// 读取opLog
	oplog, err := o.db.findOneBlockOpLogShouldUpload(hubName)
	if err != nil {
		return 0, 0, errors.Wrap(err, "o.db.OpLog.Find ALL proto.KodoBlockFeatureSize")
	}
	if len(oplog) != proto.KodoBlockFeatureSize {
		return 0, 0, errors.Errorf("o.db.OpLog.Find ALL proto.KodoBlockFeatureSize, size %v", len(oplog))
	}
	// 找到hub当前index、offet
	var hubMeta dHubMeta
	err = o.db.HubMeta.Find(bson.M{"hub_name": hubName}).One(&hubMeta)
	if err != nil {
		return 0, 0, errors.Wrap(err, "o.db.HubMeta.Find")
	}
	feature := make([]byte, 0, proto.FeatureSize*len(oplog))
	for _, v := range oplog {
		feature = append(feature, v.Feature...)
	}
	// 上传kodo
	err = o.uploadFeatureToKodo(ctx, hubName, hubMeta.FeatureVersion, hubMeta.FeatureFileIndex, feature)
	if err != nil {
		return 0, 0, errors.Wrapf(err, "uploadFeatureToKodo %v", hubMeta)
	}
	newIndex := hubMeta.FeatureFileIndex + 1
	// 更新file_meta
	for offset, v := range oplog {
		err = o.db.FileMeta.Update(bson.M{"hub_name": v.HubName, "key": v.Key}, bson.M{"$set": bson.M{
			"update_time": time.Now(),
			"status":      FileMetaStatusOK,
			"index":       hubMeta.FeatureFileIndex,
			"offset":      offset,
		}})
		if err != nil {
			return 0, 0, errors.Wrapf(err, "o.db.FileMeta.Update %v %v", v.HubName, v.Key)
		}
	}
	// 更新hub index、offet
	err = o.db.HubMeta.Update(bson.M{"hub_name": hubName}, bson.M{"$set": bson.M{"index": newIndex}})
	if err != nil {
		return 0, 0, errors.Wrap(err, "o.db.HubMeta.Update")
	}
	// 删除oplog
	opLogIds := make([]bson.ObjectId, len(oplog))
	for i, v := range oplog {
		opLogIds[i] = v.ID
	}
	removed, err := o.db.removeMultiOpLog(opLogIds)
	if err != nil {
		return 0, 0, errors.Wrap(err, "o.db.OpLog.Remove ALL after upload kodo")
	}
	if removed != len(opLogIds) {
		xl.Panicln("o.db.OpLog.RemoveAll size mismatches")
	}
	return proto.KodoBlockFeatureSize, hubMeta.FeatureFileIndex, nil
}

func (o *opLogProcess) uploadKodo(ctx context.Context) (successedNum int, err error) {
	hubs, err := o.db.countOpLogShouldUploadUser()
	if err != nil {
		return 0, errors.Wrap(err, "uploadKodo countOpLogShouldUploadUser")
	}
	buf, _ := json.Marshal(hubs)
	if len(hubs) == 0 {
		xl.Debug("countOpLogShouldUploadUser success", string(buf))
	} else {
		xl.Info("countOpLogShouldUploadUser success", string(buf))
	}
	for {
		hasOver := true
		for _, hub := range hubs {
			if hub.Cnt >= proto.KodoBlockFeatureSize {
				xl.Info("uploadKodoOneHub start", hub.HubName)
				num, index, err := o.uploadKodoOneHub(ctx, hub.HubName)
				if err != nil {
					xl.Error("uploadKodoOneHub over", hub.HubName, num, err, index)
				} else {
					xl.Info("uploadKodoOneHub over", hub.HubName, num, err, index)
				}
				successedNum += num
				if err != nil {
					return 0, errors.Wrapf(err, "uploadKodoOneHub %v", hub.HubName)
				}
				hub.Cnt -= proto.KodoBlockFeatureSize
				hasOver = false
			}
		}
		if hasOver {
			break
		}
	}
	return successedNum, nil
}

func (o *opLogProcess) uploadKodoLoop(ctx context.Context) error {
	for {
		select {
		case <-ctx.Done():
			return ctx.Err()
		default:
		}
		start := time.Now()
		n, err := o.uploadKodo(ctx)
		if err != nil {
			xl.Error("o.uploadKodo", err)
		}
		since := time.Since(start)
		if n == 0 {
			xl.Debug("o.uploadKodo over, use time", since, "record", n)
		} else {
			xl.Info("o.uploadKodo over, use time", since, "record", n)
		}
		if n == 0 {
			// TODO:conf
			time.Sleep(time.Second * 10)
		}
	}
}

func (o *opLogProcess) Start(ctx context.Context) {
	go o.uploadKodoLoop(ctx)
	go o.processAllOpLogLoop(ctx)
	go o.monitorLoop(ctx)
}
