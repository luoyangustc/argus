package auditor

import (
	"context"
	"log"
	"testing"

	"qiniu.com/argus/cap/dao"
	"qiniu.com/argus/cap/model"
	"qiniu.com/argus/cap/sand"

	mgoutil "github.com/qiniu/db/mgoutil.v3"
	"github.com/stretchr/testify/assert"
)

var (
	ctx     = context.Background()
	mgoConf = dao.CapMgoConfig{
		IdleTimeout:  5000000000,
		MgoPoolLimit: 5,
		Mgo: mgoutil.Config{
			Host:           "127.0.0.1:27017",
			DB:             "argus_cap",
			Mode:           "strong",
			SyncTimeoutInS: 5,
		},
	}
	config = model.AuditorConfig{
		IntervalSecs:        30,
		SingleTimeoutSecs:   30,
		MaxTasksNum:         30,
		PackSize:            30,
		NoSandLimitint:      0,
		SandPercentage:      5,
		RecordReserveSecond: 30,
	}
)
var colls struct {
	Auditors mgoutil.Collection `coll:"auditors"`
	Labels   mgoutil.Collection `coll:"labels"`
	Groups   mgoutil.Collection `coll:"groups"`
}

func TestAuditor(t *testing.T) {
	sess, err := mgoutil.Open(&colls, &mgoConf.Mgo)
	sess.SetPoolLimit(mgoConf.MgoPoolLimit)
	defer sess.Close()

	auditorDAO := dao.NewAuditorInMgo(&colls.Auditors)
	assert.NoError(t, err)
	groupDAO := dao.NewGroupInMgo(&colls.Groups)
	assert.NoError(t, err)
	labelDAO := dao.NewLabelInMgo(&colls.Labels)
	assert.NoError(t, err)
	taskDao, err := dao.NewTaskDao(&mgoConf)
	assert.NoError(t, err)
	sm := sand.NewSandMixer("^" + sand.SandHeadStr)

	auditor := NewAuditor(taskDao, auditorDAO, groupDAO, labelDAO, sm, config)

	{
		a := &_Auditor{}

		modeList, err := a.getCurLabelMode("mode_pulp")
		assert.NoError(t, err)
		assert.Equal(t, []string{"pulp"}, modeList)

		modeList, err = a.getCurLabelMode("mode_terror")
		assert.NoError(t, err)
		assert.Equal(t, []string{"terror"}, modeList)

		modeList, err = a.getCurLabelMode("mode_politician")
		assert.NoError(t, err)
		assert.Equal(t, []string{"politician"}, modeList)

		modeList, err = a.getCurLabelMode("mode_pulp_terror")
		assert.NoError(t, err)
		assert.Equal(t, []string{"pulp", "terror"}, modeList)

		modeList, err = a.getCurLabelMode("mode_pulp_politician")
		assert.NoError(t, err)
		assert.Equal(t, []string{"pulp", "politician"}, modeList)

		modeList, err = a.getCurLabelMode("mode_terror_politician")
		assert.NoError(t, err)
		assert.Equal(t, []string{"terror", "politician"}, modeList)

		modeList, err = a.getCurLabelMode("mode_pulp_terror_politician")
		assert.NoError(t, err)
		assert.Equal(t, []string{"pulp", "terror", "politician"}, modeList)

		ret, err := a.getRandsBySum(20, 3)
		log.Println("rand", ret, err)
	}

	{
		// prepare data for test
		err = auditorDAO.Insert(ctx, dao.AuditorInMgo{
			AuditorID:  "222",
			Valid:      "valid",
			CurGroup:   "g4",
			AbleGroups: []string{"g4"},
			SandAllNum: 0,
			SandOKNum:  0,
		})
		assert.NoError(t, err)

		err = groupDAO.Insert(ctx, dao.GroupInMgo{
			GroupID:       "g4",
			LabelModeName: "mode_police",
			RealTimeLevel: "batch",
			Level:         "",
		})
		assert.NoError(t, err)

		err = labelDAO.Insert(ctx, dao.LabelInMgo{
			Name:       "mode_police",
			LabelTypes: []string{"classify.pulp", "classify.terror"},
			Labels: map[string][]dao.LabelTitle{
				"pulp": []dao.LabelTitle{dao.LabelTitle{Title: "normal", Selected: true}, {Title: "sexy", Selected: false}, {Title: "pulp", Selected: false}},
			},
		})
		assert.NoError(t, err)

		// test case
		_, err := auditor.GetAuditorAttr(ctx, "222")
		assert.NoError(t, err)

		// _, err = auditor.FetchTasks(ctx, "222")
		// assert.NoError(t, err)

		err = auditor.CancelTasks(ctx, "222", []string{"11111"}, "123")
		assert.NoError(t, err)

		// clean the db
		err = auditorDAO.Remove(ctx, "222")
		assert.NoError(t, err)

		err = groupDAO.Remove(ctx, "g4")
		assert.NoError(t, err)

		err = labelDAO.Remove(ctx, "mode_police")
		assert.NoError(t, err)
	}

}
