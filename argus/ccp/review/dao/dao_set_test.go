package dao

import (
	"testing"

	"gopkg.in/mgo.v2/bson"
	"qiniu.com/argus/ccp/review/enums"

	"github.com/stretchr/testify/assert"
	"qiniu.com/argus/ccp/review/model"
)

func TestSetDaoInit(t *testing.T) {
	assertion := assert.New(t)
	assertion.NotNil(SetDao)
}

func TestSetDaoDB(t *testing.T) {
	assertion := assert.New(t)

	set := &model.Set{
		SourceType: enums.SourceTypeKodo,
		Type:       enums.JobTypeBatch,
		NotifyURL:  "NotifyURL",
		SetId:      bson.NewObjectId().Hex(),
	}

	err := SetDao.Insert(nil, set)
	assertion.Nil(err)
}

func TestSetDaoQueryBySets(t *testing.T) {
	assertion := assert.New(t)
	_, err := SetDao.QueryBySets(nil, []string{
		bson.NewObjectId().Hex(),
	})
	assertion.Nil(err)
}

func TestSetFilterIsValid(t *testing.T) {
	assertion := assert.New(t)

	{
		filter := SetFilter{}
		assertion.False(filter.IsValid())
	}

	{
		filter := SetFilter{
			SetId: bson.NewObjectId().Hex(),
		}
		assertion.True(filter.IsValid())
	}

	{
		filter := SetFilter{
			Uid:        1,
			SourceType: enums.SourceTypeKodo,
			Type:       enums.JobTypeStream,
		}

		assertion.True(filter.IsValid())
	}

	{
		filter := SetFilter{
			Uid:        0,
			SourceType: enums.SourceTypeKodo,
			Type:       enums.JobTypeStream,
		}

		assertion.False(filter.IsValid())
	}

	{
		filter := SetFilter{
			Uid:        1,
			SourceType: enums.SourceType("invalid"),
			Type:       enums.JobTypeStream,
		}

		assertion.False(filter.IsValid())
	}

	{
		filter := SetFilter{
			Uid:        1,
			SourceType: enums.SourceTypeKodo,
			Type:       enums.JobType("invalid"),
		}

		assertion.False(filter.IsValid())
	}
}

func TestSetFilterGetEntryDao(t *testing.T) {
	assertion := assert.New(t)

	set := &model.Set{
		SetId:      bson.NewObjectId().Hex(),
		SourceType: enums.SourceTypeKodo,
		Type:       enums.JobTypeBatch,
	}
	EntrySetCache.MustSet(set)

	{
		filter := &SetFilter{
			SetId: set.SetId,
		}

		entryDao, err := filter.GetEntryDao()
		assertion.Nil(err)
		assertion.Equal(entryDao, getEntryDao(set))
	}
}

func TestGetSetIds(t *testing.T) {
	assertion := assert.New(t)

	{
		filter := &SetFilter{
			SetId: bson.NewObjectId().Hex(),
		}

		ids, err := filter.GetSetIds()

		assertion.Nil(err)
		assertion.Equal(ids, []string{filter.SetId})
	}

	{
		filter := &SetFilter{
			Type:       enums.JobTypeBatch,
			SourceType: enums.SourceTypeKodo,
		}

		_, err := filter.GetSetIds()

		assertion.Nil(err)
	}

	{
		filter := &SetFilter{
			Type:       enums.JobTypeBatch,
			SourceType: enums.SourceTypeApi,
		}

		_, err := filter.GetSetIds()

		assertion.Nil(err)
	}
}
