package concerns

import (
	"context"
	"encoding/json"
	"testing"

	"github.com/stretchr/testify/assert"
	"gopkg.in/mgo.v2/bson"
	"qiniu.com/argus/ccp/review/dao"
	"qiniu.com/argus/ccp/review/enums"
	"qiniu.com/argus/ccp/review/model"
)

func TestEntryCounterCheck(t *testing.T) {
	assertion := assert.New(t)

	// ignore if suggestion is pass
	{
		passScene, _ := json.Marshal(model.OriginalSuggestionResultV1{
			Suggestion: enums.SuggestionPass,
		})
		entry := &model.Entry{
			SetId:    bson.NewObjectId().Hex(),
			MimeType: enums.MimeTypeImage,
			Original: &model.OriginalSuggestion{
				Suggestion: enums.SuggestionPass,
				Scenes: map[enums.Scene]interface{}{
					enums.ScenePulp: passScene,
				},
			},
			Version: "v1",
		}

		EntryCounter.CheckEntry(context.Background(), entry)

		_, err := dao.SetCounterDAO.Find(context.Background(), entry.SetId)
		assertion.Equal(err, dao.ErrNotFound)
	}

	// should increase if suggestion is review
	{
		reviewScene, _ := json.Marshal(model.OriginalSuggestionResultV1{
			Suggestion: enums.SuggestionPass,
		})
		entry := &model.Entry{
			SetId:    bson.NewObjectId().Hex(),
			MimeType: enums.MimeTypeImage,
			Original: &model.OriginalSuggestion{
				Suggestion: enums.SuggestionReview,
				Scenes: map[enums.Scene]interface{}{
					enums.ScenePulp:       reviewScene,
					enums.SceneTerror:     reviewScene,
					enums.ScenePolitician: reviewScene,
				},
			},
			Version: "v1",
		}

		dao.EntrySetCache.MustSet(&model.Set{
			SetId: entry.SetId,
		})

		EntryCounter.CheckEntry(context.Background(), entry)

		// counter, err := dao.SetCounterDAO.Find(context.Background(), entry.SetId)
		// assertion.Nil(err)
		// assertion.Equal(1, counter.Values[enums.ScenePulp])
		// assertion.Equal(1, counter.Values[enums.SceneTerror])
		// assertion.Equal(1, counter.Values[enums.ScenePolitician])

		// video type
		entry.MimeType = enums.MimeTypeVideo
		EntryCounter.CheckEntry(context.Background(), entry)

		// counter, err = dao.SetCounterDAO.Find(context.Background(), entry.SetId)
		// assertion.Nil(err)
		// assertion.Equal(1, counter.Values2[enums.ScenePulp])
		// assertion.Equal(1, counter.Values2[enums.SceneTerror])
		// assertion.Equal(1, counter.Values2[enums.ScenePolitician])

		// reduce counter
		entry.MimeType = enums.MimeTypeImage
		EntryCounter.CheckEntry(context.Background(), entry)

		// counter, err = dao.SetCounterDAO.Find(context.Background(), entry.SetId)
		// assertion.Nil(err)
		// assertion.Equal(0, counter.Values[enums.ScenePulp])
		// assertion.Equal(0, counter.Values[enums.SceneTerror])
		// assertion.Equal(0, counter.Values[enums.ScenePolitician])

		entry.MimeType = enums.MimeTypeVideo
		EntryCounter.CheckEntry(context.Background(), entry)
		EntryCounter.CheckEntries(context.Background(), []*model.Entry{entry})
		blockScene, _ := json.Marshal(model.OriginalSuggestionResultV1{
			Suggestion: enums.SuggestionBlock,
			Score:      0.8,
		})
		entry.Original = &model.OriginalSuggestion{
			Suggestion: enums.SuggestionBlock,
			Scenes: map[enums.Scene]interface{}{
				enums.ScenePulp: blockScene,
			},
		}
		EntryCounter.CheckEntry(context.Background(), entry)

		// counter, err = dao.SetCounterDAO.Find(context.Background(), entry.SetId)
		// assertion.Nil(err)
		// assertion.Equal(0, counter.Values2[enums.ScenePulp])
		// assertion.Equal(0, counter.Values2[enums.SceneTerror])
		// assertion.Equal(0, counter.Values2[enums.ScenePolitician])
	}
}
