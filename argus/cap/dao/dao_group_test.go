package dao

import (
	"testing"

	mgoutil "github.com/qiniu/db/mgoutil.v3"
	"github.com/stretchr/testify/assert"
)

func TestGroupInsert(t *testing.T) {
	sess, err := mgoutil.Open(&colls, &mgoConf.Mgo)
	sess.SetPoolLimit(mgoConf.MgoPoolLimit)
	defer sess.Close()

	groupDAO := NewGroupInMgo(&colls.Groups)
	assert.NoError(t, err)

	// clean the db first
	err = groupDAO.Remove(ctx, "g3")
	if err != nil {
		print("group does not exist, db clean")
	}

	// test case: insert
	err = groupDAO.Insert(ctx, GroupInMgo{
		GroupID:       "g3",
		LabelModeName: "mode_police",
		RealTimeLevel: "batch",
		Level:         "",
	})
	assert.NoError(t, err)

	// test case: query by id
	group, err := groupDAO.QueryByGID(ctx, "g3")
	assert.NoError(t, err)

	// test case: update
	group.LabelModeName = "mode_farmer"
	err = groupDAO.Update(ctx, group)
	assert.NoError(t, err)

	// test case: delete by name
	err = groupDAO.Remove(ctx, "g3")
	assert.NoError(t, err)
}

func TestGroupQueryAll(t *testing.T) {
	sess, err := mgoutil.Open(&colls, &mgoConf.Mgo)
	sess.SetPoolLimit(mgoConf.MgoPoolLimit)
	defer sess.Close()

	groupDAO := NewGroupInMgo(&colls.Groups)
	assert.NoError(t, err)

	_, err = groupDAO.QueryAll(ctx)
	assert.NoError(t, err)
}
