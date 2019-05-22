package facec

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"gopkg.in/mgo.v2"

	"qiniupkg.com/qiniutest/httptest.v1"
	"qiniupkg.com/x/mockhttp.v7"

	"github.com/qiniu/db/mgoutil.v3"
	"github.com/qiniu/http/restrpc.v1"
	"github.com/qiniu/xlog.v1"

	"qiniu.com/argus/argus/facec/client"
	"qiniu.com/argus/argus/facec/db"
)

const TEST_DB = "argus_test2"

func init() {
	db.Init(&mgoutil.Config{
		Host: "mongodb://127.0.0.1:27017",
		DB:   TEST_DB,
	})
}

type tLog struct {
	t *testing.T
}

func newTlog(t *testing.T) *tLog {
	return &tLog{t}
}

func (t *tLog) Write(b []byte) (int, error) {
	t.t.Logf("%s", string(b))
	return len(b), nil
}

func getMockCtx(t *testing.T) httptest.Context {
	hosts := client.Hosts{}
	xlog.SetOutputLevel(0)
	xlog.SetOutput(newTlog(t))
	svr := &Service{}
	if hosts.FacexCluster != "" {
		svr.cl = client.New(client.Config{Hosts: hosts})
	} else {
		svr.cl = &mockCl{}
	}
	featureTaskDao, err := db.NewFeatureTaskDao()
	assert.Nil(t, err)
	imgDao, err := db.NewImageDao()
	assert.Nil(t, err)
	faceDao, err := db.NewFaceDao()
	assert.Nil(t, err)
	groupDao, err := db.NewGroupDao()
	clusterTaskDao, err := db.NewClusterTaskDao()
	assert.Nil(t, err)
	aliasDao, err := db.NewAliasDao()
	assert.Nil(t, err)
	dataVersionDao, err := db.NewDataVersionDao()
	assert.Nil(t, err)
	svr.dFeatureTask = featureTaskDao
	svr.dImage = imgDao
	svr.dFace = faceDao
	svr.dGroup = groupDao
	svr.dClusterTask = clusterTaskDao
	svr.dAlias = aliasDao
	svr.dVersion = dataVersionDao
	svr.groupMutex = NewGroupMutex(dataVersionDao)

	svr.cfg = &Config{
		UseMock: true,
	}
	transport := mockhttp.NewTransport()
	router := restrpc.Router{
		PatternPrefix: "/v1",
		Mux:           restrpc.NewServeMux(),
	}
	transport.ListenAndServe("argus.ataraxia.ai.local", router.Register(svr))
	ctx := httptest.New(t)
	ctx.SetTransport(transport)
	return ctx
}

func cleanDB(t *testing.T) {
	session, err := mgo.Dial("mongodb://127.0.0.1:27017")
	assert.Nil(t, err)
	err = session.DB(TEST_DB).DropDatabase()
	assert.Nil(t, err)
}
