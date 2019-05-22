package service

import (
	"context"
	"log"
	"testing"

	mgoutil "github.com/qiniu/db/mgoutil.v3"
	"github.com/stretchr/testify/assert"
	"gopkg.in/mgo.v2/bson"
	"qbox.us/errors"
	"qbox.us/qconf/qconfapi"
	"qiniu.com/argus/ccp/manager"
	"qiniu.com/argus/ccp/manager/client"
	"qiniu.com/argus/ccp/manager/proto"
	"qiniu.com/argus/ccp/manager/proto/kodo"
	authstub "qiniu.com/auth/authstub.v1"
)

func TestRuleService(t *testing.T) {

	var (
		kodoSrcDAO proto.SrcDAO
		ruleDAO    proto.RuleDAO

		innerCfg  client.InnerConfig
		pfopRules client.PfopRules
		bjobs     client.Bjobs
	)

	var (
		colls struct {
			KodoSrc mgoutil.Collection `coll:"kodosrc"`
			Rules   mgoutil.Collection `coll:"rules"`
		}
	)

	{
		sess, err := mgoutil.Open(&colls,
			&mgoutil.Config{DB: "CCP_UT"},
		)
		if err != nil {
			log.Fatal("open mongo failed:", errors.Detail(err))
		}
		sess.SetPoolLimit(100)
		defer sess.Close()

		_, _ = colls.KodoSrc.RemoveAll(bson.M{})
		_, _ = colls.Rules.RemoveAll(bson.M{})

		kodoSrcDAO = kodo.NewKodoSrcDAO(&colls.KodoSrc)
		srcDAOMap := make(map[string]proto.SrcDAO)
		srcDAOMap[proto.SRC_KODO] = kodoSrcDAO
		ruleDAO = proto.NewRuleDAO(&colls.Rules, srcDAOMap)
	}
	{
		innerCfg = client.NewInnerConfig(&client.Saver{}, "")
		pfopRules = client.NewPfopRules(
			innerCfg, &qconfapi.Config{}, "")
		bjobs = client.NewBJobs(innerCfg, "")
	}
	var (
		rulesMng = manager.NewRules(ruleDAO, pfopRules, bjobs, nil, nil)
		ruleServ = NewRuleService(rulesMng)
	)

	rls, err := ruleServ.GetRules(context.Background(), &struct {
		Type       *string  `json:"type"`
		Status     *string  `json:"status"`
		SourceType *string  `json:"source_type"`
		SourceJSON *string  `json:"source"`
		OnlyLast   bool     `json:"only_last"`
		MimeTypes  []string `json:"mime_types"`
	}{}, &authstub.Env{})

	t.Log(rls)
	assert.NoError(t, err)

	rl, err := ruleServ.GetRules_(context.Background(), &struct {
		CmdArgs []string
	}{
		[]string{"rule_001"},
	}, &authstub.Env{})

	t.Log(rl)
	assert.Error(t, err)
}
