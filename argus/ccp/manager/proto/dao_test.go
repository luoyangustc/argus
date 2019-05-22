package proto

import (
	"context"
	"encoding/json"
	"testing"
	"time"

	mgoutil "github.com/qiniu/db/mgoutil.v3"
	"github.com/stretchr/testify/assert"
	"gopkg.in/mgo.v2/bson"
)

func TestDaoRule(t *testing.T) {
	var (
		ctx   = context.Background()
		colls struct {
			RulesColl mgoutil.Collection `coll:"rules"`
		}
	)

	{
		mgoConf := &mgoutil.Config{DB: "CCP_UT"}
		sess, err := mgoutil.Open(&colls, mgoConf)
		assert.NoError(t, err)
		sess.SetPoolLimit(100)
		defer sess.Close()

		_, _ = colls.RulesColl.RemoveAll(bson.M{})
	}

	var mockSrcDAO SrcDAO = &MockSrcDAO{}
	srcDAOMap := make(map[string]SrcDAO)
	srcDAOMap["MOCK"] = mockSrcDAO

	ruleDAO := NewRuleDAO(&colls.RulesColl, srcDAOMap)

	uid := uint32(12345)
	ruleID := "rule001"

	actionMock := struct {
		Disable   bool                   `json:"disable"`
		Threshold map[string]interface{} `json:"threshold"`
		Pipeline  string                 `json:"pipeline"`
	}{
		Disable:  true,
		Pipeline: "pipeline for mock",
	}

	actionRaw, _ := json.Marshal(actionMock)

	rule := Rule{
		RuleID:     ruleID,
		UID:        uid,
		SourceType: "MOCK",
		Source:     nil,
		Status:     RULE_STATUS_ON,
		Type:       TYPE_BATCH,
		Action:     actionRaw,
	}
	err := ruleDAO.Create(ctx, &rule)
	assert.NoError(t, err)

	err = ruleDAO.Close(ctx, uid, ruleID, time.Now())
	assert.NoError(t, err)

	t.Log(ruleID)
	ruleRet, err := ruleDAO.QueryByRuleID(ctx, uid, ruleID)
	assert.NoError(t, err)
	assert.Equal(t, ruleRet.RuleID, ruleID)
	assert.Equal(t, ruleRet.Status, RULE_STATUS_OFF)
}
