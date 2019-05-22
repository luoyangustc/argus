package model

import (
	"encoding/json"
	"fmt"

	"strings"

	"gopkg.in/mgo.v2/bson"
	"qiniu.com/argus/ccp/review/enums"
)

type Entry struct {
	ID       bson.ObjectId  `bson:"_id" json:"id"`
	SetId    string         `bson:"set_id" json:"set_id"` // For different sets
	URIGet   string         `bson:"uri_get" json:"uri_get,omitempty"`
	MimeType enums.MimeType `bson:"mimetype" json:"mimetype"`

	Original *OriginalSuggestion `bson:"original" json:"original"`
	Final    *FininalSuggestion  `bson:"final" json:"final"`
	// IMAGE: 推理结果
	// VIDEO: 各帧推理结果
	VideoCuts []*VideoCut `bson:"-" json:"video_cuts"`
	CoverUri  string      `bson:"cover_uri" json:"cover_uri"`

	CreatedAt int64  `bson:"created_at" json:"created_at"`
	Version   string `bson:"version" json:"version,omitempty"`
}

func (this *Entry) Patch() (err error) {
	if this.Version == "" {
		this.Version = "1.0"
	}
	if this.Version == "1.0" {
		for scene, v1RawJson := range this.Original.Scenes {
			var v1Scene OriginalSuggestionResultV1
			err = UnmarshalFromDBJson(v1RawJson, &v1Scene)
			if err != nil {
				return
			}
			this.Original.Scenes[scene] = v1Scene.toV2()
		}
		this.Version = "2.0"
	}
	return
}

func (this *Entry) GetAttentionScenes() (scenes []enums.Scene, err error) {
	switch this.Version {
	case "1.0":
		for scene, v1RawJson := range this.Original.Scenes {
			var v1Scene OriginalSuggestionResultV1
			err = UnmarshalFromDBJson(v1RawJson, &v1Scene)
			if v1Scene.Suggestion.IsAttention() {
				scenes = append(scenes, scene)
			}
		}

	case "2.0":
		for scene, v2RawJson := range this.Original.Scenes {
			var v2Scene OriginalSuggestionResultV2
			err = UnmarshalFromDBJson(v2RawJson, &v2Scene)
			if v2Scene.Suggestion.IsAttention() {
				scenes = append(scenes, scene)
			}
		}
	}
	return
}

func (this *Entry) GetVideoCuts() (ret []*VideoCut) {
	defer func() {
		this.VideoCuts = nil
	}()

	if this.MimeType != enums.MimeTypeVideo || this.VideoCuts == nil {
		return nil
	}

	if this.Original.Suggestion != enums.SuggestionPass {
		ret = this.GetUnNoramlVideoCuts()
	}

	if ret != nil {
		this.CoverUri = ret[0].Uri
		return
	}

	if len(this.VideoCuts) > 0 {
		this.CoverUri = this.VideoCuts[0].Uri
	}

	return
}

func (this *Entry) GetUnNoramlVideoCuts() []*VideoCut {
	var ret []*VideoCut

	entryId := this.ID.Hex()
	for _, cut := range this.VideoCuts {
		if cut.Original.Suggestion != enums.SuggestionPass {
			cut.EntryId = entryId
			ret = append(ret, cut)
		}
	}

	if len(ret) == 0 {
		return nil
	}

	return ret
}

type OriginalSuggestion struct {
	Source     string                      `bson:"source" json:"source"`
	Suggestion enums.Suggestion            `bson:"suggestion" json:"suggestion"`
	Scenes     map[enums.Scene]interface{} `bson:"scenes" json:"scenes"`
}

type OriginalSuggestionResultV1 struct {
	Suggestion enums.Suggestion `bson:"suggestion" json:"suggestion"`
	Score      float32          `bson:"score" json:"score"`
}

func UnmarshalFromDBJson(from interface{}, to interface{}) error {
	j, err := json.Marshal(from)
	if err != nil {
		return err
	}
	err = json.Unmarshal(j, to)
	return err
}

func (v1 *OriginalSuggestionResultV1) toV2() (v2 OriginalSuggestionResultV2) {
	v2.Suggestion = v1.Suggestion
	v2.Labels = make([]LabelInfoV2, 0)
	v2.Labels = append(v2.Labels, LabelInfoV2{
		Score: v1.Score,
	})
	return
}

type OriginalSuggestionResultV2 struct {
	Suggestion enums.Suggestion `bson:"suggestion" json:"suggestion"`
	Labels     []LabelInfoV2    `bson:"labels" json:"labels"`
}

type LabelInfoV2 struct {
	Label string   `bson:"label" json:"label"`
	Score float32  `bson:"score" json:"score"`
	Group string   `bson:"group" json:"group"`
	Pts   [][2]int `bson:"pts" json:"pts"`
}

type FininalSuggestion struct {
	Suggestion enums.Suggestion                 `bson:"suggestion" json:"suggestion"`
	Scenes     map[enums.Scene]enums.Suggestion `bson:"scenes" json:"scenes"`
}

type Set struct {
	ID  bson.ObjectId `bson:"_id" json:"id"`
	Uid uint32        `bson:"uid" json:"uid"`

	SetId      string           `bson:"set_id" json:"set_id"` // rule id
	SourceType enums.SourceType `bson:"source_type" json:"source_type"`
	Type       enums.JobType    `bson:"type" json:"type"`

	Automatic bool `bson:"automatic" json:"automatic"`
	Manual    bool `bson:"manual" json:"manual"`

	Bucket string `bson:"bucket" json:"bucket"`
	Prefix string `bson:"prefix" json:"prefix"`

	NotifyURL string `bson:"notify_url" json:"notify_url,omitempty"`
}

func (this *Set) ResourceId() string {
	return fmt.Sprintf("%s_%s_%s_%s", this.SourceType, this.Type, this.Bucket, this.Prefix)
}

func (this *Set) IsValid() bool {
	return this.Uid != 0 && this.SetId != "" &&
		this.SourceType.IsValid() &&
		this.Type.IsValid() &&
		(this.Bucket != "" || this.SourceType != enums.SourceTypeKodo) &&
		(this.Automatic || this.Manual)
}

type SetCounter struct {
	ID bson.ObjectId `bson:"_id" json:"-"`

	UserId     uint32 `bson:"uid" json:"-"`
	ResourceId string `bson:"resource_id" json:"-"`

	SetId string `bson:"set_id" json:"set_id"`

	// image counter
	Values     map[enums.Scene]int `bson:"values" json:"image_values"`
	LelfValues map[enums.Scene]int `bson:"left_values" json:"left_image_values"`

	// video counter
	Values2     map[enums.Scene]int `bson:"values2" json:"video_values"`
	LelfValues2 map[enums.Scene]int `bson:"left_values2" json:"left_video_values"`
	Version     int                 `bson:"version" json:"-"`
}

func (this *SetCounter) IsStreamType() bool {
	return strings.Contains(this.ResourceId, string(enums.JobTypeStream))
}

func (this *SetCounter) MergeWith(target *SetCounter) {
	for _, k := range enums.Scenes {
		if this.Values != nil && target.Values != nil {
			this.Values[k] += target.Values[k]
		}

		if this.LelfValues != nil && target.LelfValues != nil {
			this.LelfValues[k] += target.LelfValues[k]
		}

		if this.Values2 != nil && target.Values2 != nil {
			this.Values2[k] += target.Values2[k]
		}

		if this.LelfValues2 != nil && target.LelfValues2 != nil {
			this.LelfValues2[k] += target.LelfValues2[k]
		}
	}
}

type BatchEntryJob struct {
	ID     bson.ObjectId `bson:"_id"`
	Uid    uint32        `bson:"uid"`
	Bucket string        `bson:"bucket"`
	Key    string        `bson:"key"`

	SetId      string                    `bson:"set_id"`
	Status     enums.BatchEntryJobStatus `bson:"status"`
	PLineNumer int64                     `bson:"line_number"`
	StartAt    int64                     `bson:"start_at"`
}

func NewBatchEntryJobs(uid uint32, bucket, setId string, keys []string) []*BatchEntryJob {
	ret := make([]*BatchEntryJob, len(keys))

	for i, key := range keys {
		ret[i] = NewBatchEntryJob(uid, bucket, setId, key)
	}

	return ret
}

func NewBatchEntryJob(uid uint32, bucket, setId, key string) *BatchEntryJob {
	return &BatchEntryJob{
		ID:     bson.NewObjectId(),
		Uid:    uid,
		Bucket: bucket,
		Key:    key,
		SetId:  setId,
	}
}

type NotifyAlert struct {
	SourceType enums.SourceType `json:"type"`

	SetId string `json:"set_id"`

	Bucket string `json:"bucket,omitempty"`
	URIGet string `json:"uri_get,omitempty"`

	From enums.Suggestion `json:"from"`
	To   enums.Suggestion `json:"to"`
}

func NewNotifyAlert(stype enums.SourceType, setId, bucket, uriGet string, from, to enums.Suggestion) *NotifyAlert {
	return &NotifyAlert{
		SourceType: stype,
		SetId:      setId,
		Bucket:     bucket,
		URIGet:     uriGet,
		From:       from,
		To:         to,
	}
}

type VideoCut struct {
	ID       bson.ObjectId       `bson:"_id" json:"id"`
	EntryId  string              `bson:"entry_id" json:"-"`
	Uri      string              `bson:"uri" json:"uri"`
	Offset   int64               `bson:"offset" json:"offset"`
	Original *OriginalSuggestion `bson:"original" json:"original"`
}
