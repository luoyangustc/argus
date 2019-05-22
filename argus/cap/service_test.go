package cap

import (
	"log"
	"testing"

	mgoutil "github.com/qiniu/db/mgoutil.v3"
	"qbox.us/errors"

	"qiniu.com/argus/cap/auditor"
	"qiniu.com/argus/cap/dao"
	"qiniu.com/argus/cap/model"
	"qiniu.com/argus/cap/sand"
)

func TestService(t *testing.T) {

	var (
		colls struct {
			Labels   mgoutil.Collection `coll:"labels"`
			Groups   mgoutil.Collection `coll:"groups"`
			Auditors mgoutil.Collection `coll:"auditors"`
		}
	)

	mgoConf := &mgoutil.Config{DB: "CAP_UT"}
	sess, err := mgoutil.Open(&colls, mgoConf)
	if err != nil {
		log.Fatal("open mongo failed:", errors.Detail(err))
	}
	sess.SetPoolLimit(100)
	defer sess.Close()

	labelDAO := dao.NewLabelInMgo(&colls.Labels)
	groupDAO := dao.NewGroupInMgo(&colls.Groups)
	auditorDAO := dao.NewAuditorInMgo(&colls.Auditors)
	taskDAO, err := dao.NewTaskDao(&dao.CapMgoConfig{})
	if err != nil {
		log.Fatal("init task db failed:", errors.Detail(err))
	}

	sandMixer := sand.NewSandMixer("^" + sand.SandHeadStr)

	auditorHandler := auditor.NewAuditor(taskDAO, auditorDAO, groupDAO, labelDAO, sandMixer, model.AuditorConfig{})

	auditService, err := NewAuditService(auditorHandler)
	t.Log(auditService, err)

	sandService, err := NewSandService(sandMixer)
	t.Log(sandService, err)

}
