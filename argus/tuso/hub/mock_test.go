package hub

import (
	"context"
	"os"
	"testing"

	"github.com/qiniu/db/mgoutil.v3"
	xlog "github.com/qiniu/xlog.v1"
	"github.com/stretchr/testify/assert"
)

type testWriter struct {
	t *testing.T
}

func (w *testWriter) Write(b []byte) (int, error) {
	w.t.Log(string(b))
	return len(b), nil
}

func setUpTestEnvLog(t *testing.T) (cancel func()) {
	xlog.SetOutput(&testWriter{t})
	return func() {
		xlog.SetOutput(os.Stderr)
	}
}

type testEnv struct {
	s   *server
	ins *internalServer
	o   *opLogProcess
	m   *mockImageFeatureAPI
}

func prepareTestEnv(t *testing.T) testEnv {
	a := assert.New(t)
	sess, err := mgoutil.Dail("localhost", "", 0)
	a.Nil(err)
	db := new(db)
	s := &server{
		db: db,
	}
	ins := &internalServer{
		db: db,
	}
	testDB := sess.DB("toso_test")
	a.Nil(testDB.DropDatabase())
	mgoutil.InitCollections(db, testDB)
	db.createIndex()
	m := new(mockImageFeatureAPI)
	o := &opLogProcess{
		db:             db,
		api:            m,
		concurrencyNum: 5,
		uploader:       new(kodoUploaderMock),
		batchSize:      100000,
	}
	return testEnv{
		s:   s,
		ins: ins,
		o:   o,
		m:   m,
	}
}

type kodoUploaderMock struct {
	cfg KodoConfig
}

func (k *kodoUploaderMock) upload(ctx context.Context, key string, buf []byte) error {
	return nil
}
