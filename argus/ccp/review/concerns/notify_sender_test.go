package concerns

import (
	"context"
	"testing"

	"gopkg.in/mgo.v2/bson"
	"qiniu.com/argus/ccp/review/dao"
	"qiniu.com/argus/ccp/review/enums"
	"qiniu.com/argus/ccp/review/model"
)

func TestNotifySender(t *testing.T) {
	setId := bson.NewObjectId().Hex()

	set := &model.Set{
		SetId:      setId,
		SourceType: enums.SourceTypeKodo,
		Type:       enums.JobTypeBatch,
		NotifyURL:  "NotifyURL",
	}
	dao.EntrySetCache.MustSet(set)

	entry := &model.Entry{
		SetId:    setId,
		MimeType: enums.MimeTypeImage,
		Original: &model.OriginalSuggestion{
			Suggestion: enums.SuggestionPass,
			Scenes:     map[enums.Scene]interface{}{},
		},
		Final: &model.FininalSuggestion{
			Suggestion: enums.SuggestionPass,
			Scenes:     map[enums.Scene]enums.Suggestion{},
		},
	}
	ns := NewNotifySender()
	ns.Perform(context.Background(), entry)
}
