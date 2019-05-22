package service

import (
	"bytes"
	"context"
	"io"
	"log"
	"testing"

	mgoutil "github.com/qiniu/db/mgoutil.v3"
	restrpc "github.com/qiniu/http/restrpc.v1"
	"github.com/stretchr/testify/assert"
	"gopkg.in/mgo.v2/bson"
	"qbox.us/errors"
	"qbox.us/qconf/qconfapi"
	"qiniu.com/argus/ccp/manager"
	"qiniu.com/argus/ccp/manager/client"
	"qiniu.com/argus/ccp/manager/proto"
	"qiniu.com/argus/ccp/manager/proto/kodo"
)

func TestMsgService(t *testing.T) {

	var (
		kodoSrcDAO proto.SrcDAO
		ruleDAO    proto.RuleDAO

		innerCfg  client.InnerConfig
		pfopRules client.PfopRules
		bjobs     client.Bjobs
	)

	var (
		colls struct {
			KodoSrc mgoutil.Collection `coll:"kodosrc"`
			Rules   mgoutil.Collection `coll:"rules"`
		}
	)

	{
		sess, err := mgoutil.Open(&colls,
			&mgoutil.Config{DB: "CCP_UT"},
		)
		if err != nil {
			log.Fatal("open mongo failed:", errors.Detail(err))
		}
		sess.SetPoolLimit(100)
		defer sess.Close()

		_, _ = colls.KodoSrc.RemoveAll(bson.M{})
		_, _ = colls.Rules.RemoveAll(bson.M{})

		kodoSrcDAO = kodo.NewKodoSrcDAO(&colls.KodoSrc)
		srcDAOMap := make(map[string]proto.SrcDAO)
		srcDAOMap[proto.SRC_KODO] = kodoSrcDAO
		ruleDAO = proto.NewRuleDAO(&colls.Rules, srcDAOMap)
	}
	{
		innerCfg = client.NewInnerConfig(&client.Saver{}, "")
		pfopRules = client.NewPfopRules(
			innerCfg, &qconfapi.Config{}, "")
		bjobs = client.NewBJobs(innerCfg, "")
	}
	var (
		rulesMng = manager.NewRules(ruleDAO, pfopRules, bjobs, nil, nil)
		msgServ  = NewMsgService(rulesMng, nil, nil, nil, nil, nil)
	)

	err := msgServ.PostMsgPfop___(context.Background(), &struct {
		CmdArgs []string // UID/RuleID/MimeType
		ReqBody io.Reader
	}{
		CmdArgs: []string{"123", "id_001", "image"},
		ReqBody: bytes.NewBuffer(make([]byte, 0)),
	}, &restrpc.Env{})
	assert.Error(t, err)

	err = msgServ.PostMsgBjob__(context.Background(), &struct {
		CmdArgs []string // UID/RuleID
		ReqBody io.Reader
	}{
		CmdArgs: []string{"123", "id_001"},
		ReqBody: bytes.NewBuffer(make([]byte, 0)),
	}, &restrpc.Env{})
	assert.Error(t, err)

	err = msgServ.PostMsgManualStream__(context.Background(), &struct {
		CmdArgs []string // UID/RuleID
		ReqBody io.Reader
	}{
		CmdArgs: []string{"123", "id_001"},
		ReqBody: bytes.NewBuffer(make([]byte, 0)),
	}, &restrpc.Env{})
	assert.NoError(t, err)

	err = msgServ.PostMsgManualBatch__(context.Background(), &struct {
		CmdArgs []string // UID/RuleID
		ReqBody io.Reader
	}{
		CmdArgs: []string{"123", "id_001"},
		ReqBody: bytes.NewBuffer(make([]byte, 0)),
	}, &restrpc.Env{})
	assert.Error(t, err)

	err = msgServ.PostMsgReview__(context.Background(), &struct {
		CmdArgs []string // UID/RuleID
		ReqBody io.Reader
	}{
		CmdArgs: []string{"123", "id_001"},
		ReqBody: bytes.NewBuffer(make([]byte, 0)),
	}, &restrpc.Env{})
	assert.Error(t, err)

}
