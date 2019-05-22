package dao

import (
	"context"
	"testing"

	mgoutil "github.com/qiniu/db/mgoutil.v3"
	"github.com/stretchr/testify/assert"
)

var (
	ctx     = context.Background()
	mgoConf = CapMgoConfig{
		IdleTimeout:  5000000000,
		MgoPoolLimit: 5,
		Mgo: mgoutil.Config{
			Host:           "127.0.0.1:27017",
			DB:             "argus_cap",
			Mode:           "strong",
			SyncTimeoutInS: 5,
		},
	}
)
var colls struct {
	Auditors mgoutil.Collection `coll:"auditors"`
	Labels   mgoutil.Collection `coll:"labels"`
	Groups   mgoutil.Collection `coll:"groups"`
}

func TestAuditorCRUD(t *testing.T) {
	sess, err := mgoutil.Open(&colls, &mgoConf.Mgo)
	sess.SetPoolLimit(mgoConf.MgoPoolLimit)
	defer sess.Close()

	auditorDAO := NewAuditorInMgo(&colls.Auditors)
	assert.NoError(t, err)

	// clean the db first
	err = auditorDAO.Remove(ctx, "111")
	if err != nil {
		print("user 111 does not exist.")
	}

	// test case: insert
	err = auditorDAO.Insert(ctx, AuditorInMgo{
		AuditorID:  "111",
		Valid:      "valid",
		CurGroup:   "g0",
		AbleGroups: []string{"g0", "g1"},
		SandAllNum: 0,
		SandOKNum:  0,
	})
	assert.NoError(t, err)

	// test case: query by id
	auditor, err := auditorDAO.QueryByAID(ctx, "111")
	assert.NoError(t, err)

	// test case: update
	auditor.CurGroup = "g2"
	err = auditorDAO.Update(ctx, auditor)
	assert.NoError(t, err)

	// test case: delete by id
	err = auditorDAO.Remove(ctx, "111")
	assert.NoError(t, err)
}

func TestAuditorQueryAll(t *testing.T) {
	sess, err := mgoutil.Open(&colls, &mgoConf.Mgo)
	sess.SetPoolLimit(mgoConf.MgoPoolLimit)
	defer sess.Close()

	auditorDAO := NewAuditorInMgo(&colls.Auditors)
	assert.NoError(t, err)

	_, err = auditorDAO.QueryAll(ctx)
	assert.NoError(t, err)
}
