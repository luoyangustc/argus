package manager

import (
	"context"
	"encoding/json"
	"fmt"
	"time"

	xlog "github.com/qiniu/xlog.v1"
	"qiniu.com/argus/ccp/manager/client"
	"qiniu.com/argus/ccp/manager/proto"
)

type Rules interface {
	// delAll, 标示删除之前所有规则
	Set(ctx context.Context, rule *proto.Rule, delAll bool) (*proto.Rule, error)
	Del(ctx context.Context, uid uint32, ruleID string) error
	DelAll(ctx context.Context, uid uint32, srcType string, srcID string) error

	// 根据RuleID精确查找
	QueryByRuleID(ctx context.Context, uid uint32, ruleID string,
	) (*proto.Rule, error)

	// 根据SrcID查找
	QueryBySrcID(ctx context.Context, uid uint32,
		srcType string, srcID string,
		status *string,
	) ([]proto.Rule, error)

	Query(ctx context.Context, uid uint32,
		srcType *string, srcRaw json.RawMessage, // 根据srcRaw条件精确查找
		onlyLast bool, // 根据srcType查询时生效
		status *string, batchType *string, // 两张表中都有的过滤条件
		mimeTypes []string,
	) ([]proto.Rule, error)
}

func NewRules(ruleDAO proto.RuleDAO,
	pfopRules client.PfopRules, bjobs client.Bjobs,
	manualJobs client.ManualJobs, reviewCli client.ReviewClient) Rules {
	return &_Rules{
		RuleDAO:      ruleDAO,
		PfopRules:    pfopRules,
		Bjobs:        bjobs,
		ManualJobs:   manualJobs,
		ReviewClient: reviewCli,
	}
}

type _Rules struct {
	proto.RuleDAO
	client.PfopRules
	client.Bjobs
	client.ManualJobs
	client.ReviewClient
}

func (rs _Rules) Set(ctx context.Context, rule *proto.Rule, delAll bool) (*proto.Rule, error) {
	xl := xlog.FromContextSafe(ctx)
	xl.Infof("Rules.Set, %+v, %+v", rule, delAll)

	// Create init
	rule.Status = proto.RULE_STATUS_ON // ON
	rule.CreateSec = time.Now().Unix()
	rule.EndSec = 0

	// Clear Old
	if delAll {
		err := rs.DelAll(ctx, rule.UID, rule.SourceType, rule.SourceID)
		if err != nil {
			xl.Errorf("err, %+v", err)
			return nil, err
		}
	}

	// Check Overlap
	ols := rs.RuleDAO.QueryOverlap(ctx, rule)
	if len(ols) > 0 {
		err := fmt.Errorf("Overlap rules: %+v", ols)
		xl.Errorf("err, %+v", err)
		return nil, err
	}

	// 复核，先创建ReviewSet
	if rule.Review.IsOn {
		err := rs.ReviewClient.NewSet(ctx, rule)
		if err != nil {
			xl.Errorf("ReviewClient.NewSet err, %+v", err)
			return nil, err
		}
		xl.Infof("ReviewClient.NewSet success, %s", rule.RuleID)
	}

	// 人审
	if rule.Manual.IsOn {
		err := rs.ManualJobs.NewSet(ctx, rule)
		if err != nil {
			xl.Errorf("ManualJobs.NewSet err, %+v", err)
			return nil, err
		}
		xl.Infof("ManualJobs.NewSet success, %s", rule.RuleID)
	}

	// 机审
	if rule.Automatic.IsOn {
		// KODO
		if rule.SourceType == proto.SRC_KODO {
			if rule.Type == proto.TYPE_BATCH {
				// 存量走Bjob
				bjobID, err := rs.Bjobs.New(ctx, rule)
				if err != nil {
					xl.Errorf("Bjobs.New err, %+v", err)
					return nil, err
				}

				// 保存JobID
				rule.Automatic.JobID = &bjobID
				xl.Infof("Bjobs.New success, %s", rule.RuleID)
			} else {
				// 增量走Pfop
				// 补全PfopName
				err := rs.PfopRules.FillPfopName(ctx, rule)
				if err != nil {
					xl.Errorf("PfopRules.FillPfopName err, %+v", err)
					return nil, err
				}
				// 注册
				err = rs.PfopRules.Set(ctx, rule)
				if err != nil {
					xl.Errorf("PfopRules.Set err, %+v", err)
					return nil, err
				}
				xl.Infof("PfopRules.New success, %s", rule.RuleID)
			}
		}
	}

	// Create
	err := rs.RuleDAO.Create(ctx, rule)
	if err != nil {
		xl.Errorf("err, %+v", err)
		return nil, err
	}

	return rule, nil
}

func (rs _Rules) Del(ctx context.Context, uid uint32, ruleID string) error {
	xl := xlog.FromContextSafe(ctx)
	xl.Infof("Rules.Del, %d, %s", uid, ruleID)

	rule, err := rs.RuleDAO.QueryByRuleID(ctx, uid, ruleID)
	if err != nil {
		xl.Errorf("err, %+v", err)
		return err
	}

	err = rs.delRule(ctx, rule)
	if err != nil {
		xl.Errorf("err, %+v", err)
		return err
	}

	return nil
}

func (rs _Rules) DelAll(ctx context.Context, uid uint32,
	srcType string, srcID string) error {
	xl := xlog.FromContextSafe(ctx)
	xl.Infof("Rules.DelAll, %d, %s", uid, srcID)

	status := proto.RULE_STATUS_ON
	rulesOn, err := rs.RuleDAO.QueryBySrcID(ctx, uid, srcType, srcID, &status)
	if err == nil {
		for _, r := range rulesOn {
			err = rs.delRule(ctx, &r)
			if err != nil {
				xl.Errorf("err, %+v", err)
				return err
			}
		}
	}

	return nil
}

// 根据RuleID精确查找
func (rs _Rules) QueryByRuleID(ctx context.Context, uid uint32, ruleID string,
) (*proto.Rule, error) {
	return rs.RuleDAO.QueryByRuleID(ctx, uid, ruleID)
}

// 根据SrcID查找
func (rs _Rules) QueryBySrcID(ctx context.Context, uid uint32,
	srcType string, srcID string,
	status *string,
) ([]proto.Rule, error) {
	return rs.RuleDAO.QueryBySrcID(ctx, uid, srcType, srcID, status)
}

func (rs _Rules) Query(ctx context.Context, uid uint32,
	srcType *string, srcRaw json.RawMessage, // 根据srcRaw条件精确查找
	onlyLast bool, // 根据srcType查询时生效
	status *string, batchType *string, // 两张表中都有的过滤条件
	mimeTypes []string,
) ([]proto.Rule, error) {
	return rs.RuleDAO.Query(ctx, uid, srcType, srcRaw, onlyLast, status, batchType, mimeTypes)
}

//================================================================

func (rs _Rules) delRule(ctx context.Context, rule *proto.Rule) error {
	xl := xlog.FromContextSafe(ctx)
	xl.Infof("delRule, %+v", rule)

	// 机审
	if rule.Automatic.IsOn {
		// Kodo
		if rule.SourceType == proto.SRC_KODO {
			if rule.Type == proto.TYPE_BATCH {
				// 增量走Bjob
				err := rs.Bjobs.Cancel(ctx, rule.UID, rule.RuleID)
				if err != nil {
					xl.Errorf("Bjobs.Cancel err, %+v", err)
					return err
				}
			} else {
				// 增量走Pfop
				err := rs.PfopRules.Del(ctx, rule)
				if err != nil {
					xl.Errorf("Rules.Del err, %+v", err)
					return err
				}
			}
		}
	}

	err := rs.RuleDAO.Close(ctx, rule.UID, rule.RuleID, time.Now())
	if err != nil {
		xl.Errorf("err, %+v", err)
		return err
	}

	return nil
}
