package concerns

import (
	"os"
	"testing"
	"time"

	mgoutil "github.com/qiniu/db/mgoutil.v3"
	xlog "github.com/qiniu/xlog.v1"
	"qbox.us/qconf/qconfapi"
	"qiniu.com/argus/ccp/review/dao"
	"qiniupkg.com/api.v7/kodo"
)

func TestMain(m *testing.M) {
	cfg := &mgoutil.Config{
		DB:   "ccp_review_set",
		Mode: "Strong",
	}
	sess, err := dao.SetUp(cfg)

	if err != nil {
		panic(err.Error())
	}
	defer sess.Close()

	// init EntryCounterCacher
	EntryCounterCacher = NewEntryCounterCacher(
		xlog.NewWith("main"),
		5*time.Minute,
		1024,
	)

	code := m.Run()
	_ = sess.DB(cfg.DB).DropDatabase()
	os.Exit(code)
}

func TestKodoClient(m *testing.T) {
	NewKodoClient("domain", &qconfapi.Config{}, &kodo.Config{})
}
