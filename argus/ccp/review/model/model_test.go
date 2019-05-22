package model

import (
	"encoding/json"
	"testing"

	"gopkg.in/mgo.v2/bson"
	"qiniu.com/argus/ccp/review/enums"

	"github.com/stretchr/testify/assert"
)

func TestEntryModel(t *testing.T) {
	assertion := assert.New(t)

	data := `{
		"set_id": "set_id",
		"uri_get": "uri_get",
		"mimetype": "IMAGE",
		"original": {
			"suggestion": "PASS",
			"scenes": {
				"pulp": {
					"suggestion": "PASS"
				}
			}
		},
		"created_at": 1,
		"version": "v1.0"
	}`

	var entry Entry

	err := json.Unmarshal([]byte(data), &entry)
	assertion.Nil(err)

	assertion.Equal(entry.Version, "v1.0")
	assertion.Equal(entry.CreatedAt, int64(1))
	assertion.Equal(entry.Original.Suggestion, enums.SuggestionPass)
}

func TestSetResoureId(t *testing.T) {
	assertion := assert.New(t)

	set := &Set{
		SourceType: enums.SourceTypeKodo,
		Type:       enums.JobTypeStream,
		Bucket:     "bucket",
		Prefix:     "prefix",
	}

	assertion.Equal("KODO_STREAM_bucket_prefix", set.ResourceId())
}

func TestSetIsValid(t *testing.T) {
	assertion := assert.New(t)

	{
		set := &Set{
			SourceType: enums.SourceTypeKodo,
			Type:       enums.JobTypeStream,
			Uid:        1,
			SetId:      "set_id",
			Bucket:     "bucket",
			Automatic:  true,
		}

		assertion.True(set.IsValid())
	}

	{
		set := &Set{
			SourceType: enums.SourceTypeKodo,
			Type:       enums.JobTypeStream,
			SetId:      "set_id",
			Bucket:     "bucket",
			Automatic:  true,
		}

		assertion.False(set.IsValid())
	}

	{
		set := &Set{
			SourceType: enums.SourceTypeKodo,
			Type:       enums.JobTypeStream,
			Uid:        1,
			Automatic:  true,
		}

		assertion.False(set.IsValid())
	}

	{
		set := &Set{
			Type:      enums.JobTypeStream,
			Uid:       1,
			SetId:     "set_id",
			Automatic: true,
		}

		assertion.False(set.IsValid())
	}

	{
		set := &Set{
			SourceType: enums.SourceTypeKodo,
			Uid:        1,
			Automatic:  true,
			SetId:      "set_id",
		}

		assertion.False(set.IsValid())
	}

	{
		set := &Set{
			Type:       enums.JobTypeStream,
			SourceType: enums.SourceTypeKodo,
			Uid:        1,
			SetId:      "set_id",
		}

		assertion.False(set.IsValid())
	}
}

func TestSetCounterIsStreamType(t *testing.T) {
	c := &SetCounter{
		ResourceId: "xxx_STREAM_xx",
	}

	assertion := assert.New(t)
	assertion.True(c.IsStreamType())

	c.ResourceId = "xxx_BATCH_xx"
	assertion.False(c.IsStreamType())
}

func TestSetCounterMergeWith(t *testing.T) {
	c := &SetCounter{
		Values: map[enums.Scene]int{
			enums.ScenePulp:       1,
			enums.SceneTerror:     1,
			enums.ScenePolitician: 1,
		},

		LelfValues: map[enums.Scene]int{
			enums.ScenePulp:       1,
			enums.SceneTerror:     1,
			enums.ScenePolitician: 1,
		},

		Values2: map[enums.Scene]int{
			enums.ScenePulp:       1,
			enums.SceneTerror:     1,
			enums.ScenePolitician: 1,
		},

		LelfValues2: map[enums.Scene]int{
			enums.ScenePulp:       1,
			enums.SceneTerror:     1,
			enums.ScenePolitician: 1,
		},
	}

	c2 := &SetCounter{
		Values: map[enums.Scene]int{
			enums.ScenePulp:       1,
			enums.SceneTerror:     1,
			enums.ScenePolitician: 1,
		},
		LelfValues: map[enums.Scene]int{
			enums.ScenePulp:       1,
			enums.SceneTerror:     1,
			enums.ScenePolitician: 1,
		},

		Values2: map[enums.Scene]int{
			enums.ScenePulp:       1,
			enums.SceneTerror:     1,
			enums.ScenePolitician: 1,
		},

		LelfValues2: map[enums.Scene]int{
			enums.ScenePulp:       1,
			enums.SceneTerror:     1,
			enums.ScenePolitician: 1,
		},
	}

	c2.MergeWith(c)

	assertion := assert.New(t)
	assertion.Equal(2, c2.Values[enums.ScenePulp])
	assertion.Equal(2, c2.Values[enums.SceneTerror])
	assertion.Equal(2, c2.Values[enums.ScenePolitician])

	assertion.Equal(2, c2.LelfValues[enums.ScenePulp])
	assertion.Equal(2, c2.LelfValues[enums.SceneTerror])
	assertion.Equal(2, c2.LelfValues[enums.ScenePolitician])

	assertion.Equal(2, c2.Values2[enums.ScenePulp])
	assertion.Equal(2, c2.Values2[enums.SceneTerror])
	assertion.Equal(2, c2.Values2[enums.ScenePolitician])

	assertion.Equal(2, c2.LelfValues2[enums.ScenePulp])
	assertion.Equal(2, c2.LelfValues2[enums.SceneTerror])
	assertion.Equal(2, c2.LelfValues2[enums.ScenePolitician])
}

func TestEntryGetVideoCuts(t *testing.T) {
	assertion := assert.New(t)

	{
		entry := &Entry{
			MimeType: enums.MimeTypeImage,
			VideoCuts: []*VideoCut{
				&VideoCut{},
			},
			Original: &OriginalSuggestion{
				Suggestion: enums.SuggestionPass,
			},
		}
		assertion.Nil(entry.GetVideoCuts())
	}

	{
		entry := &Entry{
			MimeType: enums.MimeTypeVideo,
			Original: &OriginalSuggestion{
				Suggestion: enums.SuggestionPass,
			},
		}
		assertion.Nil(entry.GetVideoCuts())
	}

	{
		entry := &Entry{
			MimeType:  enums.MimeTypeVideo,
			VideoCuts: []*VideoCut{},
			Original: &OriginalSuggestion{
				Suggestion: enums.SuggestionPass,
			},
		}
		assertion.Nil(entry.GetVideoCuts())
	}

	{
		entry := &Entry{
			ID:       bson.NewObjectId(),
			MimeType: enums.MimeTypeVideo,
			VideoCuts: []*VideoCut{
				&VideoCut{
					ID:  bson.NewObjectId(),
					Uri: "a",
					Original: &OriginalSuggestion{
						Suggestion: enums.SuggestionReview,
					},
				},
				&VideoCut{
					ID:  bson.NewObjectId(),
					Uri: "b",
					Original: &OriginalSuggestion{
						Suggestion: enums.SuggestionReview,
					},
				},
			},
			Original: &OriginalSuggestion{
				Suggestion: enums.SuggestionReview,
			},
		}

		cuts := entry.GetVideoCuts()

		assertion.NotNil(cuts)
		assertion.Equal(2, len(cuts))

		assertion.Equal(cuts[0].EntryId, entry.ID.Hex())
		assertion.Equal(cuts[1].EntryId, entry.ID.Hex())

		assertion.Equal("a", entry.CoverUri)
	}

	{
		entry := &Entry{
			ID:       bson.NewObjectId(),
			MimeType: enums.MimeTypeVideo,
			VideoCuts: []*VideoCut{
				&VideoCut{
					ID:  bson.NewObjectId(),
					Uri: "uri",
					Original: &OriginalSuggestion{
						Suggestion: enums.SuggestionPass,
					},
				},
				&VideoCut{
					ID: bson.NewObjectId(),
					Original: &OriginalSuggestion{
						Suggestion: enums.SuggestionPass,
					},
				},
			},
			Original: &OriginalSuggestion{
				Suggestion: enums.SuggestionPass,
			},
		}

		cuts := entry.GetVideoCuts()

		assertion.Nil(cuts)
		assertion.NotEmpty(entry.CoverUri)
	}
}

func TestBatchEntryJob(t *testing.T) {
	assertion := assert.New(t)
	assertion.Len(NewBatchEntryJobs(0, "bucket", "set_id", []string{"key1", "key2"}), 2)
}

func TestNewNotifyAlert(t *testing.T) {
	assertion := assert.New(t)
	alert := NewNotifyAlert(
		enums.SourceTypeKodo,
		bson.NewObjectId().Hex(),
		"", "",
		enums.SuggestionPass,
		enums.SuggestionBlock,
	)

	assertion.NotNil(alert)
	assertion.Equal(alert.From, enums.SuggestionPass)
	assertion.Equal(alert.To, enums.SuggestionBlock)
}

func TestEntryPatching(t *testing.T) {
	v1String := []byte(`{
		"set_id" : "6f440d5e-9798-4cc0-8ba0-544e9a6f5fdb",
		"uri_get" : "qiniu://z0/argus-bcp-test/ts_ccp02/none-201807121528.jpeg",
		"mimetype" : "IMAGE",
		"original" : {
			"source" : "AUTOMATIC",
			"suggestion" : "PASS",
			"scenes" : {
				"pulp" : {
					"suggestion" : "PASS",
					"score" : 0.998207807540894
				},
				"terror" : {
					"suggestion" : "PASS",
					"score" : 0.998281478881836
				},
				"politician" : {
					"suggestion" : "PASS",
					"score" : 0.999998986721039
				}
			}
		},
		"final" : null,
		"cover_uri" : "",
		"created_at" : 1533782776,
		"version" : "1.0"
	}`)
	entry := Entry{}
	assertion := assert.New(t)
	assertion.Nil(json.Unmarshal(v1String, &entry))
	assertion.Equal("1.0", entry.Version)
	assertion.Nil(entry.Patch())
	assertion.Equal("2.0", entry.Version)
}

func TestNewBatchEntryJob(t *testing.T) {
	assertion := assert.New(t)
	setId := "setId"
	bucket := "bucket"

	keys := []string{"a", "b"}
	assertion.NotEmpty(NewBatchEntryJobs(0, bucket, setId, keys))
	assertion.NotEmpty(NewBatchEntryJob(0, bucket, setId, keys[0]))
}
