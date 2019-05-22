package kodo

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"

	mgoutil "github.com/qiniu/db/mgoutil.v3"
	xlog "github.com/qiniu/xlog.v1"
	mgo "gopkg.in/mgo.v2"
	"gopkg.in/mgo.v2/bson"
	"qiniu.com/argus/ccp/manager/proto"
)

// Implement proto.SrcDAO
type KodoSrcDAO struct {
	srcColl *mgoutil.Collection
}

func NewKodoSrcDAO(srcColl *mgoutil.Collection) proto.SrcDAO {

	_ = srcColl.EnsureIndex(mgo.Index{Key: []string{"source_id"}, Unique: true})

	return &KodoSrcDAO{
		srcColl: srcColl,
	}
}

func (src *KodoSrcDAO) Create(ctx context.Context, rule *proto.Rule) error {
	xl := xlog.FromContextSafe(ctx)
	srcColl := src.srcColl.CopySession()
	defer srcColl.CloseSession()

	if rule == nil {
		err := fmt.Errorf("rule nil")
		xl.Errorf("%+v", err)
		return err
	}

	kodoSrcInMgo := KodoSrcInMgo{}
	err := convertRuleToKodoSrc(rule, &kodoSrcInMgo)
	if err != nil {
		xl.Errorf("convert err, %+v", err)
		return err
	}

	err = srcColl.Insert(kodoSrcInMgo)
	if err != nil {
		xl.Errorf("Src.Insert err, %+v", err)
		return err
	}

	return nil
}

func (src *KodoSrcDAO) QueryBySrcID(ctx context.Context, uid uint32, srcID string,
) (*proto.Rule, error) {
	xl := xlog.FromContextSafe(ctx)
	srcColl := src.srcColl.CopySession()
	defer srcColl.CloseSession()

	var (
		kodoSrcInMgo KodoSrcInMgo
		query        = bson.M{
			"uid":       uid,
			"source_id": srcID,
		}
	)

	err := srcColl.Find(query).One(&kodoSrcInMgo)
	if err != nil {
		xl.Errorf("Src.Find err, %+v", err)
		return nil, err
	}

	rule := proto.Rule{}
	err = convertKodoSrcToRule(&kodoSrcInMgo, &rule)
	if err != nil {
		xl.Errorf("convert err, %+v", err)
		return nil, err
	}

	return &rule, nil
}

func (src *KodoSrcDAO) Query(ctx context.Context,
	uid uint32, srcRaw json.RawMessage, // 根据srcRaw条件精确查找
	batchType *string, // 两张表中都有的过滤条件
) ([]proto.Rule, error) {
	xl := xlog.FromContextSafe(ctx)
	srcColl := src.srcColl.CopySession()
	defer srcColl.CloseSession()

	var (
		kodoSrcInMgoArr []KodoSrcInMgo
		query           = bson.M{
			"uid": uid,
		}
	)

	if len(srcRaw) > 0 {
		// get info from srcRaw
		kodoSrc := KodoSrc{}
		err := json.Unmarshal(srcRaw, &kodoSrc)
		if err != nil {
			xl.Errorf("convert err, %+v", err)
			return nil, err
		}
		if len(kodoSrc.Buckets) > 0 { // <= 0 为查询UID下所有
			query["bucket"] = kodoSrc.Buckets[0].Bucket
			if kodoSrc.Buckets[0].Prefix != nil {
				query["prefix"] = *kodoSrc.Buckets[0].Prefix
			}
		}
	}
	if batchType != nil {
		query["type"] = *batchType
	}

	// Find
	err := srcColl.Find(query).All(&kodoSrcInMgoArr)
	if err != nil {
		xl.Errorf("Src.Find err, %+v", err)
		return nil, err
	}

	rules := []proto.Rule{}
	for _, srcInMgo := range kodoSrcInMgoArr {
		rule := proto.Rule{}
		err = convertKodoSrcToRule(&srcInMgo, &rule)
		if err != nil {
			xl.Errorf("convert err, %+v", err)
			return nil, err
		}

		rules = append(rules, rule)
	}

	return rules, nil
}

func (src *KodoSrcDAO) QueryOverlap(ctx context.Context,
	uid uint32, srcRaw json.RawMessage,
	batchType *string) []string {
	xl := xlog.FromContextSafe(ctx)
	srcColl := src.srcColl.CopySession()
	defer srcColl.CloseSession()

	var (
		pfx             = ""
		kodoSrcInMgoArr []KodoSrcInMgo
		query           = bson.M{
			"uid": uid,
		}
	)

	if len(srcRaw) > 0 {
		// get info from srcRaw
		kodoSrc := KodoSrc{}
		err := json.Unmarshal(srcRaw, &kodoSrc)
		if err != nil {
			xl.Errorf("convert err, %+v", err)
			return nil
		}
		if len(kodoSrc.Buckets) > 0 { // <= 0 为查询UID下所有
			query["bucket"] = kodoSrc.Buckets[0].Bucket
			if kodoSrc.Buckets[0].Prefix != nil {
				pfx = *kodoSrc.Buckets[0].Prefix
			}
		}

	}
	if batchType != nil {
		query["type"] = *batchType
	}

	// Find
	err := srcColl.Find(query).All(&kodoSrcInMgoArr)
	if err != nil {
		xl.Errorf("Src.Find err, %+v", err)
		return nil
	}

	// check Overlap
	var srcIDs []string
	for _, ks := range kodoSrcInMgoArr {

		ksPfx := ""
		if ks.Prefix != nil {
			ksPfx = *ks.Prefix
		}

		var isOl bool
		if len(pfx) > len(ksPfx) {
			isOl = strings.HasPrefix(pfx, ksPfx)
		} else {
			isOl = strings.HasPrefix(ksPfx, pfx)
		}

		if isOl {
			srcIDs = append(srcIDs, ks.SourceID)
		}
	}

	return srcIDs
}
