package concerns

import (
	"bufio"
	"context"
	"encoding/json"
	"os"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"qiniu.com/argus/ccp/review/model"
)

func TestBatchEntryJobProcessor(t *testing.T) {
	jobProcessor := NewBatchEntryJobProcessor(context.Background(), &KodoClient{}, 1)
	jobProcessor.Start()
	jobProcessor.Close()
	// assertion := assert.New(t)

	// qconfJson := `{"access_key":"4_odedBxmrAHiu4Y0Qp0HPG0NANCf6VAsAjWL_k9","mc_rw_timeout_ms":100,"lc_chan_bufsize":16000,"master_hosts":["http://10.200.20.25:8510"],"lc_expires_ms":300000,"lc_duration_ms":5000,"mc_hosts":["10.200.20.23:11211"],"secret_key":""}`

	// var qconf qconfapi.Config
	// err := json.Unmarshal([]byte(qconfJson), &qconf)
	// assertion.Nil(err)

	// kconfJson := `{"RSHost":"http://10.200.20.25:12501","RSFHost":"http://10.200.20.25:12501","IoHost":"10.200.20.23","UpHosts":["http://10.200.20.23:3710"]}`
	// var kConf kodo.Config
	// err = json.Unmarshal([]byte(kconfJson), &kConf)
	// assertion.Nil(err)

	// jobProcessor := NewBatchEntryJobProcessor(context.Background(),
	// 	NewKodoClient(
	// 		"http://10.200.20.23:12500", &qconf, &kConf), 5)

	// set := &model.Set{
	// 	SetId:      bson.NewObjectId().Hex(),
	// 	SourceType: enums.SourceTypeKodo,
	// 	Type:       enums.JobTypeBatch,
	// 	Automatic:  true,
	// }

	// err = dao.SetDao.Insert(jobProcessor.ctx, set)
	// assertion.Nil(err)

	// job := model.NewBatchEntryJob(
	// 	1380538984,
	// 	"argus-bcp",
	// 	set.SetId,
	// 	"local_test_rule_153129465720180711153852_CCP_TEST/5b45b3c289ca37000624abfb/20180711T153822_20180711T153851__9",
	// )

	// err = dao.BatchEntryJobDAO.BatchInsert(jobProcessor.ctx, []*model.BatchEntryJob{
	// 	job,
	// })
	// assertion.Nil(err)

	// err = dao.BatchEntryJobDAO.StartJob(jobProcessor.ctx, job.ID)
	// assertion.Nil(err)

	// jobProcessor.processJob(job)

	// _, err = dao.BatchEntryJobDAO.Find(jobProcessor.ctx, job.ID.Hex())
	// assertion.Nil(err)
}

func TestKeyIterReadLine(t *testing.T) {
	assertion := assert.New(t)

	file, err := os.Open("./batch_entry_test.data")
	assertion.Nil(err)
	defer file.Close()

	rd := &KeyIter{
		scanner: bufio.NewReader(file),
	}

	line, hasMore := rd.ReadLine()
	assertion.True(hasMore)

	fields := strings.Split(line, "\t")

	var entry model.Entry
	err = json.Unmarshal([]byte(fields[1]), &entry)
	assertion.Nil(err)

	line, hasMore = rd.ReadLine()
	assertion.False(hasMore)
	assertion.Equal(line, "")
}
