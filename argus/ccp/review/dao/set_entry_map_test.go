package dao

import (
	"testing"

	"qiniu.com/argus/ccp/review/enums"

	"github.com/stretchr/testify/assert"
	"gopkg.in/mgo.v2/bson"
	"qiniu.com/argus/ccp/review/model"
)

func Test_SetEntryMap_MustGet(t *testing.T) {
	assertion := assert.New(t)

	set := &model.Set{
		SetId: bson.NewObjectId().Hex(),
	}

	EntrySetCache.MustSet(set)

	foundSet, err := EntrySetCache.MustGet(set.SetId)
	assertion.Nil(err)
	assertion.NotNil(foundSet)

	foundSet, err = EntrySetCache.MustGet(bson.NewObjectId().Hex())
	assertion.Equal(err, ErrNotFound)
	assertion.Nil(foundSet)
}

func Test_SetEntryMap_MustSet(t *testing.T) {
	assertion := assert.New(t)

	set := &model.Set{
		SetId: bson.NewObjectId().Hex(),
	}

	EntrySetCache.cache = make(map[string]*model.Set)

	EntrySetCache.MustSet(set)
	assertion.Equal(1, len(EntrySetCache.cache))

	set.SetId = bson.NewObjectId().Hex()
	EntrySetCache.MustSet(set)

	assertion.Equal(2, len(EntrySetCache.cache))

	// reset entry set cache limit
	entrySetCacheLimit = 1

	set.ID = bson.NewObjectId()
	EntrySetCache.MustSet(set)
	assertion.Equal(1, len(EntrySetCache.cache))

	entrySetCacheLimit = 10000 // reset the limit
}

func Test_SetEntryMap_GetDao(t *testing.T) {
	assertion := assert.New(t)
	{
		set := &model.Set{
			SetId:      bson.NewObjectId().Hex(),
			SourceType: enums.SourceTypeKodo,
			Type:       enums.JobTypeBatch,
		}

		EntrySetCache.MustSet(set)

		getDao, err := EntrySetCache.GetDao(set.SetId)
		assertion.Nil(err)
		assertion.Equal(QnInvEntriesDao, getDao)
	}

	{
		set := &model.Set{
			SetId:      bson.NewObjectId().Hex(),
			SourceType: enums.SourceTypeKodo,
			Type:       enums.JobTypeStream,
		}

		EntrySetCache.MustSet(set)

		getDao, err := EntrySetCache.GetDao(set.SetId)
		assertion.Nil(err)
		assertion.Equal(QnIncEntriesDao, getDao)
	}

	{
		set := &model.Set{
			SetId:      bson.NewObjectId().Hex(),
			SourceType: enums.SourceTypeApi,
			Type:       enums.JobTypeBatch,
		}

		EntrySetCache.MustSet(set)

		getDao, err := EntrySetCache.GetDao(set.SetId)
		assertion.Nil(err)
		assertion.Equal(ApiInvEntriesDao, getDao)
	}

	{
		set := &model.Set{
			SetId:      bson.NewObjectId().Hex(),
			SourceType: enums.SourceTypeApi,
			Type:       enums.JobTypeStream,
		}

		EntrySetCache.MustSet(set)

		getDao, err := EntrySetCache.GetDao(set.SetId)
		assertion.Nil(err)
		assertion.Equal(ApiIncEntriesDao, getDao)
	}
}
