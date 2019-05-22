package service

import (
	"context"
	"encoding/json"

	xlog "github.com/qiniu/xlog.v1"
	"qiniu.com/argus/ccp/manager"
	"qiniu.com/argus/ccp/manager/client"
	"qiniu.com/argus/ccp/manager/proto"
	authstub "qiniu.com/auth/authstub.v1"
)

type RuleService interface {
	PostRules(
		ctx context.Context,
		rule *proto.Rule,
		env *authstub.Env,
	) (*proto.Rule, error)

	PostRulesClose_(
		ctx context.Context,
		req *struct {
			CmdArgs []string // ruleID
		},
		env *authstub.Env,
	) error

	GetRules_(
		ctx context.Context,
		req *struct {
			CmdArgs []string // ruleID
		},
		env *authstub.Env,
	) (*proto.Rule, error)

	GetRules(
		ctx context.Context,
		req *struct {
			Type       *string  `json:"type"`
			Status     *string  `json:"status"`
			SourceType *string  `json:"source_type"`
			SourceJSON *string  `json:"source"` // 根据具体Src中的json字段，有则过滤
			OnlyLast   bool     `json:"only_last"`
			MimeTypes  []string `json:"mime_types"`
		},
		env *authstub.Env,
	) ([]proto.Rule, error)

	GetRulesSource__(
		ctx context.Context,
		req *struct {
			CmdArgs []string // srcType|srcID
			Status  *string  `json:"status"`
		},
		env *authstub.Env,
	) ([]proto.Rule, error)
}

func NewRuleService(rules manager.Rules) RuleService {
	return &_RuleService{
		Rules: rules,
	}
}

type _RuleService struct {
	manager.Rules
}

func (rs *_RuleService) PostRules(
	ctx context.Context,
	rule *proto.Rule,
	env *authstub.Env,
) (*proto.Rule, error) {
	xl := xlog.FromContextSafe(ctx)
	rule.UID = env.Uid
	rule.Utype = env.Utype
	xl.Infof("PostRules.rule, %d, %s", env.Uid, client.JsonStr(rule))

	retRule, err := rs.Rules.Set(ctx, rule, true)
	if err != nil {
		xl.Errorf("err, %+v", err)
		return nil, err
	}

	return retRule, nil
}

func (rs *_RuleService) PostRulesClose_(
	ctx context.Context,
	req *struct {
		CmdArgs []string // ruleID
	},
	env *authstub.Env,
) error {
	xl := xlog.FromContextSafe(ctx)
	var ruleID string
	if len(req.CmdArgs) > 0 {
		ruleID = req.CmdArgs[0]
	}
	xl.Infof("PostRulesClose_.ruleID, %d, %s", env.Uid, ruleID)

	err := rs.Rules.Del(ctx, env.Uid, ruleID)
	if err != nil {
		xl.Errorf("err, %+v", err)
		return err
	}

	return nil
}

func (rs *_RuleService) GetRules_(
	ctx context.Context,
	req *struct {
		CmdArgs []string // ruleID
	},
	env *authstub.Env,
) (*proto.Rule, error) {
	xl := xlog.FromContextSafe(ctx)
	var ruleID string
	if len(req.CmdArgs) > 0 {
		ruleID = req.CmdArgs[0]
	}

	rule, err := rs.Rules.QueryByRuleID(ctx, env.Uid, ruleID)
	if err != nil {
		xl.Errorf("err, %+v", err)
		return nil, err
	}

	return rule, nil
}

func (rs *_RuleService) GetRules(
	ctx context.Context,
	req *struct {
		Type       *string  `json:"type"`
		Status     *string  `json:"status"`
		SourceType *string  `json:"source_type"`
		SourceJSON *string  `json:"source"` // 根据具体Src中的json字段，有则过滤
		OnlyLast   bool     `json:"only_last"`
		MimeTypes  []string `json:"mime_types"`
	},
	env *authstub.Env,
) ([]proto.Rule, error) {
	xl := xlog.FromContextSafe(ctx)

	var srcRaw json.RawMessage
	if req.SourceJSON != nil {
		srcRaw = json.RawMessage(*req.SourceJSON)
	}

	rules, err := rs.Rules.Query(ctx, env.Uid,
		req.SourceType, srcRaw, req.OnlyLast, req.Status, req.Type, req.MimeTypes)
	if err != nil {
		xl.Errorf("err, %+v", err)
		return nil, err
	}

	return rules, nil
}

func (rs *_RuleService) GetRulesSource__(
	ctx context.Context,
	req *struct {
		CmdArgs []string // srcType|srcID
		Status  *string  `json:"status"`
	},
	env *authstub.Env,
) ([]proto.Rule, error) {
	xl := xlog.FromContextSafe(ctx)
	var (
		srcType string
		srcID   string
	)
	if len(req.CmdArgs) > 1 {
		srcType = req.CmdArgs[0]
		srcID = req.CmdArgs[1]
	}

	rules, err := rs.Rules.QueryBySrcID(ctx, env.Uid,
		srcType, srcID, req.Status)
	if err != nil {
		xl.Errorf("err, %+v", err)
		return nil, err
	}

	return rules, nil
}
