package concerns

import (
	"context"

	"qiniu.com/argus/ccp/review/enums"
	"qiniu.com/argus/ccp/review/model"
)

var (
	EntryCounter *_EntryCounter
)

type _EntryCounter struct{}

func (_ *_EntryCounter) CheckEntry(ctx context.Context, entry *model.Entry) {
	// just check the block and auto disable case
	if entry.Original == nil ||
		!entry.Original.Suggestion.IsAttention() {
		return
	}

	scenes, err := entry.GetAttentionScenes()
	if err == nil && len(scenes) > 0 {
		pushToEntryCounterCacher(scenes, entry.SetId, entry.MimeType)
	}
}

func (this *_EntryCounter) CheckEntries(ctx context.Context, entries []*model.Entry) {
	for _, entry := range entries {
		this.CheckEntry(ctx, entry)
	}
}

func pushToEntryCounterCacher(scenes []enums.Scene,
	setId string,
	mType enums.MimeType,
) {
	c := &model.SetCounter{
		SetId:   setId,
		Values:  make(map[enums.Scene]int),
		Values2: make(map[enums.Scene]int),
	}

	switch mType {
	case enums.MimeTypeImage:
		for _, scene := range scenes {
			c.Values[scene] = 1
		}
	case enums.MimeTypeVideo:
		for _, scene := range scenes {
			c.Values2[scene] = 1
		}
	}

	EntryCounterCacher.Add(c)
}
