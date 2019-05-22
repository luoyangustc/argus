package dao

import (
	"context"
	"testing"
	"time"

	"qiniu.com/argus/ccp/review/enums"

	"github.com/stretchr/testify/assert"
	"gopkg.in/mgo.v2/bson"
	"qiniu.com/argus/ccp/review/model"
)

func TestEntryDaoInit(t *testing.T) {
	assertion := assert.New(t)

	entryDaos := []EntryDAO{
		QnIncEntriesDao,
		QnInvEntriesDao,
		ApiIncEntriesDao,
		ApiInvEntriesDao,
	}

	for _, entryDao := range entryDaos {
		assertion.NotNil(entryDao)
	}
}

func TestEntryDaoQuery(t *testing.T) {
	assertion := assert.New(t)

	entryDaos := []EntryDAO{QnIncEntriesDao, QnInvEntriesDao, ApiIncEntriesDao, ApiInvEntriesDao}

	for _, entryDao := range entryDaos {
		_, err := entryDao.Query(context.Background(), &EntryFilter{
			SetIds: []string{bson.NewObjectId().Hex()},
		}, NewPaginator("", 25))

		assertion.Nil(err)
	}
}

func TestEntryFind(t *testing.T) {
	assertion := assert.New(t)

	entryDaos := []EntryDAO{
		QnIncEntriesDao, QnInvEntriesDao, ApiIncEntriesDao, ApiInvEntriesDao,
	}

	for _, entryDao := range entryDaos {
		_, err := entryDao.Find(context.Background(), "")
		assertion.Equal(err, ErrInvalidId)
	}
}

func TestEntryDaoInsert(t *testing.T) {
	assertion := assert.New(t)

	setId := bson.NewObjectId().Hex()
	entry := &model.Entry{
		SetId:    setId,
		URIGet:   "xxxx",
		MimeType: enums.MimeTypeImage,
		Original: &model.OriginalSuggestion{},
	}

	entryDaos := []EntryDAO{QnIncEntriesDao, QnInvEntriesDao, ApiIncEntriesDao, ApiInvEntriesDao}

	for _, entryDao := range entryDaos {
		err := entryDao.Insert(context.Background(), entry)
		assertion.Nil(err)
	}
}

func TestEntryDaoBatchInsert(t *testing.T) {
	assertion := assert.New(t)

	setId := bson.NewObjectId().Hex()

	entries := []*model.Entry{
		&model.Entry{
			ID:       bson.NewObjectId(),
			SetId:    setId,
			URIGet:   "xxxx",
			MimeType: enums.MimeTypeImage,
			Original: &model.OriginalSuggestion{},
		},
		&model.Entry{
			ID:       bson.NewObjectId(),
			SetId:    setId,
			URIGet:   "xxxx",
			MimeType: enums.MimeTypeVideo,
			Original: &model.OriginalSuggestion{},
			VideoCuts: []*model.VideoCut{
				&model.VideoCut{
					Uri:      "xxx",
					Original: &model.OriginalSuggestion{},
				},
				&model.VideoCut{
					Uri:      "xxx",
					Original: &model.OriginalSuggestion{},
				},
			},
		},
	}

	err := QnIncEntriesDao.BatchInsert(context.Background(), entries)
	assertion.Nil(err)

	_, err = VideoCutDAO.Query(context.Background(), entries[1].ID.Hex(), nil)
	assertion.Nil(err)
}

func TestEntryDaoUpdate(t *testing.T) {
	assertion := assert.New(t)

	setId := bson.NewObjectId().Hex()
	entry := &model.Entry{
		SetId:    setId,
		URIGet:   "xxxx",
		MimeType: enums.MimeTypeImage,
		Original: &model.OriginalSuggestion{
			Suggestion: enums.SuggestionBlock,
		},
		Final: &model.FininalSuggestion{
			Suggestion: enums.SuggestionBlock,
		},
	}

	entryDaos := []EntryDAO{QnIncEntriesDao, QnInvEntriesDao, ApiIncEntriesDao, ApiInvEntriesDao}

	for _, entryDao := range entryDaos {
		err := entryDao.Insert(context.Background(), entry)
		assertion.Nil(err)

		// change finial suggestion from block to pass
		entry.Final = &model.FininalSuggestion{}
		entry.Final.Suggestion = enums.SuggestionPass
		err = entryDao.Update(nil, entry.ID.Hex(), entry)
		assertion.Nil(err)
	}
}

func TestEntryFilter(t *testing.T) {
	assertion := assert.New(t)

	{
		filer := &EntryFilter{
			SetIds: []string{"1", "2"},
		}

		assertion.Equal(filer.toQueryParams(), bson.M{
			"set_id": bson.M{
				"$in": filer.SetIds,
			},
		})
	}

	{
		filer := &EntryFilter{
			SetIds:     []string{"1", "2"},
			Scene:      enums.ScenePulp,
			Suggestion: enums.SuggestionPass,
		}

		assertion.Equal(filer.toQueryParams(), bson.M{
			"set_id": bson.M{
				"$in": filer.SetIds,
			},
			"original.suggestion":             filer.Suggestion,
			"original.scenes.pulp.suggestion": filer.Suggestion,
			"final": nil,
		})
	}

	{
		filer := &EntryFilter{
			SetIds:     []string{"1", "2"},
			Scene:      enums.ScenePulp,
			Suggestion: enums.SuggestionBlock,
		}

		assertion.Equal(filer.toQueryParams(), bson.M{
			"set_id": bson.M{
				"$in": filer.SetIds,
			},
			"original.suggestion":             filer.Suggestion,
			"original.scenes.pulp.suggestion": enums.SuggestionBlock,
			"final": nil,
		})
	}

	{
		filer := &EntryFilter{
			SetIds: []string{"1", "2"},
			Scene:  enums.ScenePulp,
			Min:    60,
			Max:    90,
		}

		assertion.Equal(filer.toQueryParams(), bson.M{
			"set_id": bson.M{
				"$in": filer.SetIds,
			},
			"original.scenes.pulp.score": bson.M{
				"$gt": filer.Min,
				"$lt": filer.Max,
			},
			"final": nil,
		})
	}

	{
		filer := &EntryFilter{
			SetIds: []string{"1", "2"},

			StartAt: time.Now().Unix(),
			EndAt:   time.Now().Unix(),
		}

		assertion.Equal(filer.toQueryParams(), bson.M{
			"set_id": bson.M{
				"$in": filer.SetIds,
			},
			"created_at": bson.M{
				"$gt": filer.StartAt,
				"$lt": filer.EndAt,
			},
		})
	}
}

func TestEntryDaoRemove(t *testing.T) {
	assertion := assert.New(t)

	setId := bson.NewObjectId().Hex()
	entry := &model.Entry{
		SetId:    setId,
		URIGet:   "xxxx",
		MimeType: enums.MimeTypeImage,
		Original: &model.OriginalSuggestion{},
	}

	for _, entryDao := range []EntryDAO{
		QnIncEntriesDao, QnInvEntriesDao, ApiIncEntriesDao, ApiInvEntriesDao,
	} {
		err := entryDao.Insert(context.Background(), entry)
		assertion.Nil(err)
		err = entryDao.Remove(context.Background(), setId)
		assertion.Nil(err)
	}
}
