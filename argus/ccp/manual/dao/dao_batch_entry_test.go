package dao

import (
	"context"
	"testing"

	"qiniu.com/argus/ccp/manual/enums"

	mgoutil "github.com/qiniu/db/mgoutil.v3"
	"github.com/stretchr/testify/assert"
)

func TestBatchEntryDao(t *testing.T) {
	var (
		ctx     = context.Background()
		mgoConf = CcpCapMgoConfig{
			IdleTimeout:  5000000000,
			MgoPoolLimit: 5,
			Mgo: mgoutil.Config{
				Host:           "127.0.0.1:27017",
				DB:             "argus_cap_test",
				Mode:           "strong",
				SyncTimeoutInS: 5,
			},
		}
	)

	batchEntryDao, err := NewBatchEntryInMgo(mgoConf)
	assert.NoError(t, err)
	batch := BatchEntryInMgo{
		SetId:  "test",
		Status: enums.BatchEntryJobStatusNew,
	}
	err = batchEntryDao.BatchInsert(ctx, &batch)
	assert.NoError(t, err)

	err = batchEntryDao.StartJob(ctx, batch.SetId)
	assert.NoError(t, err)

	ups, err := batchEntryDao.QueryByID(ctx, batch.SetId)
	assert.NoError(t, err)
	assert.Equal(t, enums.BatchEntryJobStatusProcess, ups.Status)

	err = batchEntryDao.UpdateStatus(ctx, batch.SetId, enums.BatchEntryJobStatusFailed)
	assert.NoError(t, err)

	query2, err := batchEntryDao.QueryByID(ctx, batch.SetId)
	assert.NoError(t, err)
	assert.Equal(t, enums.BatchEntryJobStatusFailed, query2.Status)

	err = batchEntryDao.Remove(ctx, batch.SetId)
	assert.NoError(t, err)
}
