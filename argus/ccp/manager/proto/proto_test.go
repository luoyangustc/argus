package proto

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestProto(t *testing.T) {

	uid := uint32(12345)
	ruleID := "rule001"

	rule := Rule{
		RuleID:     ruleID,
		UID:        uid,
		SourceType: SRC_KODO,
		Source:     nil,
		Status:     RULE_STATUS_ON,
		Type:       TYPE_BATCH,
	}

	ruleInMgo := RuleInMgo{}
	err := ruleInMgo.FromRule(&rule)
	assert.NoError(t, err)
	assert.Equal(t, ruleInMgo.UID, uid)
	assert.Equal(t, ruleInMgo.RuleID, ruleID)
	assert.Equal(t, ruleInMgo.Status, RULE_STATUS_ON)
	assert.Equal(t, ruleInMgo.Type, TYPE_BATCH)

	ruleRet := Rule{}
	err = ruleInMgo.ToRule(nil, &ruleRet)
	assert.NoError(t, err)
	assert.Equal(t, ruleRet.UID, uid)
	assert.Equal(t, ruleRet.RuleID, ruleID)
	assert.Equal(t, ruleRet.Status, RULE_STATUS_ON)
	assert.Equal(t, ruleRet.Type, TYPE_BATCH)
}
