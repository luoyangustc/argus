package client

import (
	"context"
	"fmt"
	"net/http"

	"github.com/qiniu/rpc.v3"
	xlog "github.com/qiniu/xlog.v1"
	"qiniu.com/argus/ccp/manager/proto"
)

type NotifyCallback interface {
	PostNotifyMsg(ctx context.Context, rule *proto.Rule, msg interface{}) error
}

type _NotifyCallback struct {
}

func NewNotifyCallback() NotifyCallback {
	return &_NotifyCallback{}
}

func (nc *_NotifyCallback) PostNotifyMsg(ctx context.Context, rule *proto.Rule, msg interface{}) error {
	xl := xlog.FromContextSafe(ctx)

	xl.Infof("PostNotifyMsg %+v, Rule: %+v", msg, rule)

	if rule.NotifyURL == nil {
		err := fmt.Errorf("PostNotifyMsg err, NotifyURL nil, ruleID: %s", rule.RuleID)
		xl.Errorf("%+v", err)
		return err
	}

	rpcClient := rpc.Client{Client: &http.Client{}}
	err := rpcClient.CallWithJson(ctx, nil, "POST", *rule.NotifyURL, msg)
	if err != nil {
		xl.Errorf("CallWithJson err, %v", err)
		return err
	}

	return nil
}
