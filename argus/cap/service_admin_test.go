package cap

import (
	"context"
	"testing"

	mgoutil "github.com/qiniu/db/mgoutil.v3"
	restrpc "github.com/qiniu/http/restrpc.v1"
	"github.com/stretchr/testify/assert"
	"net/http/httptest"

	"qiniu.com/argus/cap/dao"
	"qiniu.com/argus/cap/model"
)

func TestPostAuditors(t *testing.T) {
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
	)
	var colls struct {
		Auditors mgoutil.Collection `coll:"auditors"`
		Labels   mgoutil.Collection `coll:"labels"`
		Groups   mgoutil.Collection `coll:"groups"`
	}

	sess, err := mgoutil.Open(&colls, &mgoConf.Mgo)
	assert.NoError(t, err)
	sess.SetPoolLimit(mgoConf.MgoPoolLimit)
	defer sess.Close()

	labelDAO := dao.NewLabelInMgo(&colls.Labels)
	groupDAO := dao.NewGroupInMgo(&colls.Groups)
	auditorDAO := dao.NewAuditorInMgo(&colls.Auditors)
	adminService, err := NewAdminService(labelDAO, groupDAO, auditorDAO)
	assert.NoError(t, err)

	env := &restrpc.Env{
		W:   httptest.NewRecorder(),
		Req: httptest.NewRequest("GET", "/v1/admin", nil),
	}

	// table `argus-cap/auditors` related api
	// clean db first in case of duplicate key
	req := struct{ CmdArgs []string }{CmdArgs: []string{"111"}}
	err = adminService.PostAuditorsDelete_(ctx, &req, env)
	if err != nil {
		println("db clean now!")
	}
	// assert.NoError(t, err)

	newAuditor := model.Auditor{
		ID:         "111",
		Valid:      "valid",
		CurGroup:   "g0",
		AbleGroups: []string{"g0", "g1"},
		SandOKNum:  0,
		SandAllNum: 0,
	}
	err = adminService.PostAuditors(ctx, &newAuditor, env)
	assert.NoError(t, err)

	_, err = adminService.GetAuditors(ctx, env)
	assert.NoError(t, err)

	_, err = adminService.GetAuditors_(ctx, &req, env)
	assert.NoError(t, err)

	err = adminService.PostAuditorsDelete_(ctx, &req, env)
	assert.NoError(t, err)

	// table `argus-cap/group` related api
	// clean db first in case of duplicate key
	req = struct{ CmdArgs []string }{CmdArgs: []string{"g3"}}
	err = adminService.PostGroupsDelete_(ctx, &req, env)
	if err != nil {
		println("db clean now!")
	}

	newGroup := model.AuditorGroup{
		GroupID:       "g3",
		Mode:          "mode_test",
		RealTimeLevel: "batch",
		Level:         "",
	}
	err = adminService.PostGroups(ctx, &newGroup, env)
	assert.NoError(t, err)

	_, err = adminService.GetGroups(ctx, env)
	assert.NoError(t, err)

	req = struct{ CmdArgs []string }{CmdArgs: []string{"g3"}}
	targetGroup, err := adminService.GetGroups_(ctx, &req, env)
	assert.NoError(t, err)
	assert.Equal(t, "g3", targetGroup.GroupID)

	err = adminService.PostGroupsDelete_(ctx, &req, env)
	assert.NoError(t, err)

	// table `argus-cap/label` related api
	// clean db first in case of duplicate key
	req = struct{ CmdArgs []string }{CmdArgs: []string{"mode_test"}}
	err = adminService.PostLabelModesDelete_(ctx, &req, env)
	if err != nil {
		println("db clean now!")
	}

	newLabel := model.LabelMode{
		Name:       "mode_test",
		LabelTypes: []string{"classify.pulp", "classify.terror"},
		Labels: map[string][]model.LabelTitle{
			"pulp": []model.LabelTitle{
				model.LabelTitle{
					Title:    "normal",
					Selected: true,
				},
			},
		},
	}
	err = adminService.PostLabelModes(ctx, &newLabel, env)
	assert.NoError(t, err)

	_, err = adminService.GetLabelModes(ctx, env)
	assert.NoError(t, err)

	req = struct{ CmdArgs []string }{CmdArgs: []string{"mode_test"}}
	targetMode, err := adminService.GetLabelModes_(ctx, &req, env)
	assert.NoError(t, err)
	assert.Equal(t, "mode_test", targetMode.Name)

	err = adminService.PostLabelModesDelete_(ctx, &req, env)
	assert.NoError(t, err)
}
