package manager

import (
	"context"
	"encoding/json"
	"log"
	"testing"

	mgoutil "github.com/qiniu/db/mgoutil.v3"
	"github.com/stretchr/testify/assert"
	"gopkg.in/mgo.v2/bson"
	"qbox.us/errors"
	"qbox.us/qconf/qconfapi"
	"qiniu.com/argus/ccp/manager/client"
	"qiniu.com/argus/ccp/manager/proto"
	"qiniu.com/argus/ccp/manager/proto/kodo"
)

func TestRules(t *testing.T) {

	var (
		ctx = context.Background()

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
		rulesMng = NewRules(ruleDAO, pfopRules, bjobs, nil, nil)
	)

	_, err := rulesMng.Set(ctx, &proto.Rule{
		RuleID: "rl_id_001",
	}, true)
	assert.Error(t, err)

	pfx := "qqq"
	kodoSrc := kodo.KodoSrc{
		Buckets: []struct {
			Bucket string  `json:"bucket"`
			Prefix *string `json:"prefix,omitempty"`
		}{
			struct {
				Bucket string  `json:"bucket"`
				Prefix *string `json:"prefix,omitempty"`
			}{
				Bucket: "argus-bcp-test",
				Prefix: &pfx,
			},
		},
	}
	srcRaw, _ := json.Marshal(kodoSrc)
	rl, err := rulesMng.Set(ctx, &proto.Rule{
		RuleID:     "rl_id_001",
		SourceType: "KODO",
		SourceID:   "src_id_001",
		Source:     srcRaw,
	}, true)
	assert.NoError(t, err)
	t.Log(rl)
	assert.Equal(t, proto.RULE_STATUS_ON, rl.Status)

	rls, err := rulesMng.QueryBySrcID(ctx, 0, "KODO", "src_id_001", nil)
	assert.NoError(t, err)
	assert.Equal(t, len(rls), 1)

	rl2, err := rulesMng.QueryByRuleID(ctx, 0, "rl_id_001")
	assert.NoError(t, err)
	assert.NotNil(t, rl2)

	rls3, err := rulesMng.Query(ctx, 0, nil, nil, true, nil, nil, nil)
	assert.NoError(t, err)
	assert.Equal(t, len(rls3), 1)

	err = rulesMng.Del(ctx, 0, "rl_id_001")
	assert.NoError(t, err)

	rl4, err := rulesMng.QueryByRuleID(ctx, 0, "rl_id_001")
	assert.NoError(t, err)
	assert.NotNil(t, rl4)
	assert.Equal(t, proto.RULE_STATUS_OFF, rl4.Status)

	err = rulesMng.DelAll(ctx, 0, "KODO", "src_id_001")
	assert.NoError(t, err)

}
