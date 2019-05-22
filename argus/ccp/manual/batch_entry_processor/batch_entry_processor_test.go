package batch_entry_processor

import (
	"context"
	"testing"

	mgoutil "github.com/qiniu/db/mgoutil.v3"
	"github.com/stretchr/testify/assert"

	"qiniu.com/argus/ccp/conf"
	"qiniu.com/argus/ccp/manual/client"
	"qiniu.com/argus/ccp/manual/dao"
	"qiniu.com/argus/ccp/manual/model"
	"qiniu.com/argus/ccp/manual/saver"
)

func TestBatchEntryJobProcessor(t *testing.T) {
	var (
		ctx  = context.Background()
		conf = conf.BatchEntryProcessorConf{
			MaxPool: 2,
			Gzip:    false,
		}
	)
	setDao, err := dao.NewSetInMgo(dao.CcpCapMgoConfig{
		IdleTimeout:  5000000000,
		MgoPoolLimit: 5,
		Mgo: mgoutil.Config{
			Host:           "127.0.0.1:27017",
			DB:             "argus_new_cap",
			Mode:           "strong",
			SyncTimeoutInS: 5,
		}})
	assert.NoError(t, err)

	batchEntryDao, err := dao.NewBatchEntryInMgo(dao.CcpCapMgoConfig{})
	assert.NoError(t, err)

	capClient := client.NewCAPClient(&model.CAPConfig{})
	bucketSaver := saver.NewBucketSaver("", nil, nil, &setDao)
	//handler := cap.NewMaunalHandler(ctx, setDao, streamEntryDao, batchEntryDao, capClient)
	jobProcessor := NewBatchEntryJobProcessor(&conf, saver.NewKodoClient("host", nil, nil),
		&setDao, &batchEntryDao, &capClient, &bucketSaver)
	jobProcessor.Start(ctx)
	jobProcessor.Close()
}
