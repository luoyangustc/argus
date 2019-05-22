package concerns

import (
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"gopkg.in/mgo.v2/bson"
	"qiniu.com/argus/ccp/review/enums"
	"qiniu.com/argus/ccp/review/model"
)

func TestEntryCounterCacherStart(t *testing.T) {
	assertion := assert.New(t)

	EntryCounterCacher.Start()

	c := &model.SetCounter{
		SetId: bson.NewObjectId().Hex(),
		Values: map[enums.Scene]int{
			enums.ScenePulp: 1,
		},
		Values2: map[enums.Scene]int{
			enums.ScenePulp: 1,
		},
	}

	EntryCounterCacher.Add(c)

	c1 := &model.SetCounter{
		SetId: c.SetId,
		Values: map[enums.Scene]int{
			enums.ScenePulp: 1,
		},
		Values2: map[enums.Scene]int{
			enums.ScenePulp: 1,
		},
	}

	EntryCounterCacher.Add(c1)

	c2 := &model.SetCounter{
		SetId: c.SetId,
		Values: map[enums.Scene]int{
			enums.ScenePulp: -1,
		},
		Values2: map[enums.Scene]int{
			enums.ScenePulp: -1,
		},
	}

	EntryCounterCacher.Add(c2)

	time.Sleep(1 * time.Second)

	assertion.Equal(1, EntryCounterCacher.get(c.SetId).Values[enums.ScenePulp])
	assertion.Equal(1, EntryCounterCacher.get(c.SetId).Values2[enums.ScenePulp])
}
