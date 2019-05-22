package proto

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"time"

	mgoutil "github.com/qiniu/db/mgoutil.v3"
	xlog "github.com/qiniu/xlog.v1"
	mgo "gopkg.in/mgo.v2"
	"gopkg.in/mgo.v2/bson"
)

type RuleDAO interface {
	Create(ctx context.Context, rule *Rule) error
	Close(ctx context.Context, uid uint32, ruleID string, endTime time.Time) error

	// 根据RuleID精确查找
	QueryByRuleID(ctx context.Context, uid uint32, ruleID string,
	) (*Rule, error)

	// 根据SrcID查找
	QueryBySrcID(ctx context.Context, uid uint32,
		srcType string, srcID string,
		status *string,
	) ([]Rule, error)

	Query(ctx context.Context, uid uint32,
		srcType *string, srcRaw json.RawMessage, // 根据srcRaw条件精确查找
		onlyLast bool, // 根据srcType查询时生效
		status *string, batchType *string, // 两张表中都有的过滤条件
		mimeTypes []string,
	) ([]Rule, error)

	// 返回冲突的RuleID
	QueryOverlap(ctx context.Context, rule *Rule) []string
}

// SrcDAO interface For Each Src, KODO/API...
type SrcDAO interface {
	Create(ctx context.Context, rule *Rule) error

	QueryBySrcID(ctx context.Context, uid uint32, srcID string,
	) (*Rule, error)

	Query(ctx context.Context,
		uid uint32, srcRaw json.RawMessage, // 根据srcRaw条件精确查找
		batchType *string, // 两张表中都有的过滤条件
	) ([]Rule, error)

	// 查询有重叠的资源ID
	QueryOverlap(ctx context.Context,
		uid uint32, srcRaw json.RawMessage,
		batchType *string) []string
}

func NewRuleDAO(rulesColl *mgoutil.Collection, srcDAOMap map[string]SrcDAO) RuleDAO {

	_ = rulesColl.EnsureIndex(mgo.Index{Key: []string{"rule_id"}, Unique: true})
	_ = rulesColl.EnsureIndex(mgo.Index{Key: []string{"source_type", "source_id"}})

	return &_RuleDAO{
		rulesColl: rulesColl,
		srcDAOMap: srcDAOMap,
	}
}

type _RuleDAO struct {
	rulesColl *mgoutil.Collection
	srcDAOMap map[string]SrcDAO
}

//================================================================

func (r *_RuleDAO) Create(ctx context.Context, rule *Rule) error {
	xl := xlog.FromContextSafe(ctx)
	if rule == nil {
		err := fmt.Errorf("rule nil")
		xl.Errorf("%+v", err)
		return err
	}

	srcDAO := r.srcDAOMap[rule.SourceType]
	if srcDAO == nil {
		err := fmt.Errorf("SrcDAO nil, %s", rule.SourceType)
		xl.Errorf("%+v", err)
		return err
	}

	// 先确保convert成功
	ruleInMgo := RuleInMgo{}
	err := ruleInMgo.FromRule(rule)
	if err != nil {
		xl.Errorf("convert err, %+v", err)
		return err
	}

	// 查询 & 录入Src
	srcFromDB, err := srcDAO.QueryBySrcID(ctx, ruleInMgo.UID, ruleInMgo.SourceID)
	if err != nil || srcFromDB == nil {
		// 不存在，则录入
		err = srcDAO.Create(ctx, rule)
		if err != nil {
			xl.Errorf("SrcDAO.Create err, %s, %+v", rule.SourceType, err)
			return err
		}
	}

	// 录入Rule
	rulesColl := r.rulesColl.CopySession()
	defer rulesColl.CloseSession()
	err = rulesColl.Insert(ruleInMgo)
	if err != nil {
		xl.Errorf("Rules.Insert err, %+v", err)
		return err
	}

	return nil
}

func (r *_RuleDAO) Close(ctx context.Context, uid uint32, ruleID string, endTime time.Time) error {
	xl := xlog.FromContextSafe(ctx)
	rulesColl := r.rulesColl.CopySession()
	defer rulesColl.CloseSession()

	var (
		query = bson.M{
			"uid":     uid,
			"rule_id": ruleID,
		}
	)

	err := rulesColl.Update(
		query,
		bson.M{
			"$set": bson.M{"status": RULE_STATUS_OFF, "end_time": endTime},
		},
	)
	if err != nil {
		xl.Errorf("Rules.Update err, %+v", err)
		return err
	}

	return nil
}

func (r *_RuleDAO) QueryByRuleID(ctx context.Context, uid uint32,
	ruleID string, // 根据RuleID精确查找
) (*Rule, error) {
	xl := xlog.FromContextSafe(ctx)

	rl, err := r.queryRuleInMgo(ctx, uid, ruleID)
	if err != nil {
		xl.Errorf("queryRuleInMgo err, %+v", err)
		return nil, err
	}

	srcDAO := r.srcDAOMap[rl.SourceType]
	if srcDAO == nil {
		err := fmt.Errorf("SrcDAO nil, %s", rl.SourceType)
		xl.Errorf("%+v", err)
		return nil, err
	}

	ruleFromSrc, err := srcDAO.QueryBySrcID(ctx, uid, rl.SourceID)
	if err != nil {
		xl.Errorf("SrcDAO.QueryByID err, %s, %+v", rl.SourceType, err)
		return nil, err
	}

	rule := Rule{}
	err = rl.ToRule(ruleFromSrc.Source, &rule)
	if err != nil {
		xl.Errorf("convert err, %+v", err)
		return nil, err
	}

	return &rule, nil
}

func (r *_RuleDAO) QueryBySrcID(ctx context.Context, uid uint32,
	srcType string, srcID string, // 根据SrcID查找
	status *string,
) ([]Rule, error) {
	xl := xlog.FromContextSafe(ctx)

	rls, err := r.queryRuleInMgoArr(ctx, uid, &srcType, &srcID, status, nil, nil)
	if err != nil {
		xl.Errorf("queryRuleInMgoArr err, %+v", err)
		return nil, err
	}

	srcDAO := r.srcDAOMap[srcType]
	if srcDAO == nil {
		err := fmt.Errorf("SrcDAO nil, %s", srcType)
		xl.Errorf("%+v", err)
		return nil, err
	}

	ruleFromSrc, err := srcDAO.QueryBySrcID(ctx, uid, srcID)
	if err != nil {
		xl.Errorf("SrcDAO.QueryByID err, %s, %+v", srcType, err)
		return nil, err
	}

	rules := []Rule{}
	for _, rl := range rls {
		rule := Rule{}
		err = rl.ToRule(ruleFromSrc.Source, &rule)
		if err != nil {
			xl.Errorf("convert err, %+v", err)
			return nil, err
		}

		rules = append(rules, rule)
	}

	return rules, nil
}

func (r *_RuleDAO) Query(ctx context.Context, uid uint32,
	srcType *string, srcRaw json.RawMessage, // 根据srcRaw条件精确查找
	onlyLast bool, // 根据srcType查询时生效
	status *string, batchType *string, // 两张表中都有的冗余条件，for filter
	mimeTypes []string,
) ([]Rule, error) {
	xl := xlog.FromContextSafe(ctx)

	if srcType != nil { // 根据 Src 过滤查找
		srcDAO := r.srcDAOMap[*srcType]
		if srcDAO == nil {
			err := fmt.Errorf("SrcDAO nil, %s", *srcType)
			xl.Errorf("%+v", err)
			return nil, err
		}

		ruleArrFromSrc, err := srcDAO.Query(ctx, uid, srcRaw, batchType)
		if err != nil {
			xl.Errorf("SrcDAO.Query err, %s, %+v", *srcType, err)
			return nil, err
		}

		rules := []Rule{}
		for _, ruleFromSrc := range ruleArrFromSrc {

			rls, err := r.queryRuleInMgoArr(ctx, uid, srcType, &ruleFromSrc.SourceID,
				status, batchType, mimeTypes)
			if err != nil {
				xl.Errorf("queryRuleInMgoArr err, %+v", err)
				return nil, err
			}

			tmpRules := []Rule{}
			for _, rl := range rls {
				rule := Rule{}
				err = rl.ToRule(ruleFromSrc.Source, &rule)
				if err != nil {
					xl.Errorf("convert err, %+v", err)
					return nil, err
				}

				tmpRules = append(tmpRules, rule)

				if onlyLast { // 只取最新一个，不继续
					break
				}
			}

			rules = append(rules, tmpRules...)
		}

		return rules, nil
	}

	// 无法根据ID精确查找，也没有根据SrcRaw查找
	// 则根据过滤条件批量查找rules
	rls, err := r.queryRuleInMgoArr(ctx, uid, nil, nil,
		status, batchType, mimeTypes)
	if err != nil {
		xl.Errorf("queryRuleInMgoArr err, %+v", err)
		return nil, err
	}

	rules := []Rule{}
	for _, rl := range rls {
		srcDAO := r.srcDAOMap[rl.SourceType]
		if srcDAO == nil {
			err := fmt.Errorf("SrcDAO nil, %s", rl.SourceType)
			xl.Errorf("%+v", err)
			return nil, err
		}

		ruleFromSrc, err := srcDAO.QueryBySrcID(ctx, uid, rl.SourceID)
		if err != nil {
			xl.Errorf("SrcDAO.QueryByID err, %s, %+v", rl.SourceType, err)
			return nil, err
		}

		rule := Rule{}
		err = rl.ToRule(ruleFromSrc.Source, &rule)
		if err != nil {
			xl.Errorf("convert err, %+v", err)
			return nil, err
		}

		rules = append(rules, rule)
	}

	return rules, nil
}

func (r *_RuleDAO) QueryOverlap(ctx context.Context, rule *Rule) []string {
	xl := xlog.FromContextSafe(ctx)
	if rule == nil {
		err := fmt.Errorf("rule nil")
		xl.Errorf("%+v", err)
		return nil
	}

	srcDAO := r.srcDAOMap[rule.SourceType]
	if srcDAO == nil {
		err := fmt.Errorf("SrcDAO nil, %s", rule.SourceType)
		xl.Errorf("%+v", err)
		return nil
	}

	var (
		rlIDs []string
		stON  = RULE_STATUS_ON
	)
	srcIDs := srcDAO.QueryOverlap(ctx, rule.UID, rule.Source, &rule.Type)
	for _, si := range srcIDs {
		rls, _ := r.queryRuleInMgoArr(ctx, rule.UID,
			&rule.SourceType, &si, &stON, &rule.Type, nil)
		for _, rl := range rls {
			rlIDs = append(rlIDs, rl.RuleID)
		}
	}

	return rlIDs
}

//================================================================

func (r *_RuleDAO) queryRuleInMgo(ctx context.Context,
	uid uint32, ruleID string,
) (*RuleInMgo, error) {
	xl := xlog.FromContextSafe(ctx)
	rulesColl := r.rulesColl.CopySession()
	defer rulesColl.CloseSession()

	var (
		ruleInMgo RuleInMgo
		query     = bson.M{
			"uid":     uid,
			"rule_id": ruleID,
		}
	)

	err := rulesColl.Find(query).One(&ruleInMgo)
	if err != nil {
		xl.Errorf("Rules.Find err, %+v", err)
		return nil, err
	}

	return &ruleInMgo, nil
}

func (r *_RuleDAO) queryRuleInMgoArr(ctx context.Context,
	uid uint32, srcType *string, srcID *string,
	status *string, batchType *string, mimeTypes []string,
) ([]RuleInMgo, error) {
	xl := xlog.FromContextSafe(ctx)
	rulesColl := r.rulesColl.CopySession()
	defer rulesColl.CloseSession()

	var (
		ruleInMgoArr []RuleInMgo
		query        = bson.M{
			"uid": uid,
		}
	)

	if srcType != nil {
		query["source_type"] = *srcType
	}
	if srcID != nil {
		query["source_id"] = *srcID
	}
	if batchType != nil {
		query["type"] = *batchType
	}
	if status != nil {
		query["status"] = *status
	}
	if mimeTypes != nil {
		for _, mt := range mimeTypes {
			query[fmt.Sprintf("%s.is_on", strings.ToLower(mt))] = true
		}
	}

	// 按创建时间倒序排序
	err := rulesColl.Find(query).Sort("-create_time").All(&ruleInMgoArr)
	if err != nil {
		xl.Errorf("Rules.Find err, %+v", err)
		return nil, err
	}

	return ruleInMgoArr, nil
}

//================================================================

// Implement SrcDAO
type MockSrcDAO struct {
	// Mock for UT
}

var _ SrcDAO = &MockSrcDAO{}

func (src *MockSrcDAO) Create(ctx context.Context, rule *Rule) error {
	return nil
}

func (src *MockSrcDAO) QueryBySrcID(ctx context.Context, uid uint32, srcID string,
) (*Rule, error) {
	return &Rule{}, nil
}

func (src *MockSrcDAO) Query(ctx context.Context,
	uid uint32, srcRaw json.RawMessage, // 根据srcRaw条件精确查找
	batchType *string, // 两张表中都有的过滤条件
) ([]Rule, error) {
	return nil, nil
}

func (src *MockSrcDAO) QueryOverlap(ctx context.Context,
	uid uint32, srcRaw json.RawMessage,
	batchType *string) []string {
	return nil
}
