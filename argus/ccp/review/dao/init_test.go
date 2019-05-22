package dao

import (
	"os"
	"testing"

	mgoutil "github.com/qiniu/db/mgoutil.v3"
)

func TestMain(m *testing.M) {
	cfg := &mgoutil.Config{
		DB:   "ccp_review_set",
		Mode: "Strong",
	}

	sess, err := SetUp(cfg)

	if err != nil {
		panic(err.Error())
	}
	defer sess.Close()

	code := m.Run()
	_ = sess.DB(cfg.DB).DropDatabase()
	os.Exit(code)
}
