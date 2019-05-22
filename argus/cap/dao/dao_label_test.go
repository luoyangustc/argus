package dao

import (
	"testing"

	mgoutil "github.com/qiniu/db/mgoutil.v3"
	"github.com/stretchr/testify/assert"
)

func TestLabelInsert(t *testing.T) {
	sess, err := mgoutil.Open(&colls, &mgoConf.Mgo)
	sess.SetPoolLimit(mgoConf.MgoPoolLimit)
	defer sess.Close()

	labelDAO := NewLabelInMgo(&colls.Labels)
	assert.NoError(t, err)

	// clean the db first
	err = labelDAO.Remove(ctx, "mode_police")
	if err != nil {
		print("label does not exist, db clean")
	}

	// test case: insert
	err = labelDAO.Insert(ctx, LabelInMgo{
		Name:       "mode_police",
		LabelTypes: []string{"classify.pulp", "classify.terror"},
		Labels: map[string][]LabelTitle{
			"pulp": []LabelTitle{LabelTitle{Title: "normal", Selected: true}, {Title: "sexy", Selected: false}, {Title: "pulp", Selected: false}},
		},
	})
	assert.NoError(t, err)

	// test case: query by name
	label, err := labelDAO.QueryByName(ctx, "mode_police")
	assert.NoError(t, err)

	// test case: update
	label.LabelTypes = []string{"classify.pulp", "classify.terror", "classify.politician"}
	err = labelDAO.Update(ctx, label)
	assert.NoError(t, err)

	// test case: delete by name
	err = labelDAO.Remove(ctx, "mode_police")
	assert.NoError(t, err)
}

func TestLabelQueryAll(t *testing.T) {
	sess, err := mgoutil.Open(&colls, &mgoConf.Mgo)
	sess.SetPoolLimit(mgoConf.MgoPoolLimit)
	defer sess.Close()

	labelDAO := NewLabelInMgo(&colls.Labels)
	assert.NoError(t, err)

	_, err = labelDAO.QueryAll(ctx)
	assert.NoError(t, err)
}
