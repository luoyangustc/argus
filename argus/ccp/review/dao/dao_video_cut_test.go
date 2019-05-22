package dao

import (
	"context"
	"testing"

	"qiniu.com/argus/ccp/review/enums"

	"github.com/stretchr/testify/assert"
	"gopkg.in/mgo.v2/bson"
	"qiniu.com/argus/ccp/review/model"
)

func TestVideoCutDaoInit(t *testing.T) {
	assertion := assert.New(t)
	assertion.NotNil(VideoCutDAO)
}

func TestVideoCutDaoInDB(t *testing.T) {
	assertion := assert.New(t)

	entryId := bson.NewObjectId().Hex()

	docs := []*model.VideoCut{
		&model.VideoCut{
			EntryId: entryId,
			Uri:     "xxx",
			Offset:  12,
			Original: &model.OriginalSuggestion{
				Suggestion: enums.SuggestionReview,
			},
		},
		&model.VideoCut{
			EntryId: entryId,
			Uri:     "xxx",
			Offset:  24,
			Original: &model.OriginalSuggestion{
				Suggestion: enums.SuggestionBlock,
			},
		},
	}

	err := VideoCutDAO.BatchInsert(context.TODO(), docs)
	assertion.Nil(err)

	foundCuts, err := VideoCutDAO.Query(context.TODO(), entryId, nil)
	assertion.Nil(err)
	assertion.NotEmpty(foundCuts)
	assertion.True(2 == len(foundCuts))
}

func TestVideoCutCount(t *testing.T) {
	assertion := assert.New(t)

	_, err := VideoCutDAO.Count(context.Background(), "123")
	assertion.Equal(ErrInvalidId, err)

	_, err = VideoCutDAO.Count(context.Background(), bson.NewObjectId().Hex())
	assertion.Nil(err)
}
