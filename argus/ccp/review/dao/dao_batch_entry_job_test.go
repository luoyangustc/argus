package dao

import (
	"context"
	"testing"

	"github.com/stretchr/testify/assert"
	"qiniu.com/argus/ccp/review/enums"
	"qiniu.com/argus/ccp/review/model"
)

func TestBatchEntryJobDAOInit(t *testing.T) {
	assertion := assert.New(t)
	assertion.NotNil(BatchEntryJobDAO)
}

func TestBatchEntryJobDaoDB_BatchInsert(t *testing.T) {
	assertion := assert.New(t)

	jobs := model.NewBatchEntryJobs(0, "bucket", "setId", []string{"a", "b"})

	{
		err := BatchEntryJobDAO.BatchInsert(context.Background(), jobs)
		assertion.Nil(err)
	}
}
func TestBatchEntryJobDaoDB_Query(t *testing.T) {
	assertion := assert.New(t)

	{
		_, err := BatchEntryJobDAO.Query(context.Background(), enums.BatchEntryJobStatusNew)
		assertion.Nil(err)
	}
}

func TestBatchEntryJobDaoDB_Remove(t *testing.T) {
	assertion := assert.New(t)

	{
		jobs := model.NewBatchEntryJobs(0, "bucket", "setId2", []string{"a", "b"})
		err := BatchEntryJobDAO.BatchInsert(context.Background(), jobs)
		assertion.Nil(err)

		queryJobs, err := BatchEntryJobDAO.Query(context.Background(), enums.BatchEntryJobStatusNew)
		if assertion.Nil(err) && assertion.NotEmpty(queryJobs) {
			num := 0
			for _, j := range queryJobs {
				if j.SetId == "setId2" && (j.Key == "a" || j.Key == "b") && j.Status == enums.BatchEntryJobStatusNew {
					num++
				}
			}
			assertion.Equal(2, num)
		}

		err = BatchEntryJobDAO.UpdateStatusBySetId(context.Background(), "setId2", enums.BatchEntryJobStatusSuccess)
		assertion.Nil(err)
		queryJobs, err = BatchEntryJobDAO.Query(context.Background(), enums.BatchEntryJobStatusSuccess)
		if assertion.Nil(err) && assertion.NotEmpty(queryJobs) {
			num := 0
			for _, j := range queryJobs {
				if j.SetId == "setId2" && (j.Key == "a" || j.Key == "b") && j.Status == enums.BatchEntryJobStatusSuccess {
					num++
				}
			}
			assertion.Equal(2, num)
		}

		{
			jobId := queryJobs[0].ID
			err = BatchEntryJobDAO.UpdateStatus(context.Background(), jobId, enums.BatchEntryJobStatusSuccess, enums.BatchEntryJobStatusNew)
			assertion.Nil(err)
			j, err := BatchEntryJobDAO.Find(context.Background(), jobId.Hex())
			assertion.Nil(err)
			assertion.Equal(j.ID.Hex(), jobId.Hex())
			assertion.Equal(enums.BatchEntryJobStatusNew, j.Status)

			err = BatchEntryJobDAO.StartJob(context.Background(), jobId)
			assertion.Nil(err)
			err = BatchEntryJobDAO.UpdateLineNumber(context.Background(), jobId, 1)
			assertion.Nil(err)

			j, err = BatchEntryJobDAO.Find(context.Background(), jobId.Hex())
			assertion.Nil(err)
			assertion.Equal(int(j.PLineNumer), 1)
			assertion.Equal(enums.BatchEntryJobStatusProcess, j.Status)
		}

		assertion.Nil(err)
		err = BatchEntryJobDAO.RemoveBySetId(context.Background(), "setId2")
		assertion.Nil(err)
	}
}
