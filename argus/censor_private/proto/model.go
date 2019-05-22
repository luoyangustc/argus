package proto

import (
	"time"

	"gopkg.in/mgo.v2/bson"
	"qiniu.com/argus/censor_private/util"
)

type OuterServiceConfig struct {
	Host    string        `json:"host"`
	Timeout time.Duration `json:"timeout"`
}

type Set struct {
	Id               string     `bson:"id" json:"id"`
	Name             string     `bson:"name" json:"name"`
	Type             SetType    `bson:"type" json:"type"`
	MonitorInterval  int        `bson:"monitor_interval,omitempty" json:"monitor_interval"`
	MimeTypes        []MimeType `bson:"mime_types" json:"mime_types"`
	Scenes           []Scene    `bson:"scenes" json:"scenes"`
	Uri              string     `bson:"uri,omitempty" json:"uri"`
	CutIntervalMsecs int        `bson:"cut_interval_msecs,omitempty" json:"cut_interval_msecs,omitempty"`
	Status           SetStatus  `bson:"status" json:"status"`
	CreatedAt        int64      `bson:"created_at" json:"created_at"`
	ModifiedAt       int64      `bson:"modified_at" json:"modified_at"`
}

type SetHistory struct {
	Id               bson.ObjectId `bson:"_id" json:"-"`
	SetId            string        `bson:"set_id" json:"-"`
	Name             string        `bson:"name" json:"name"`
	Type             SetType       `bson:"type" json:"type"`
	MonitorInterval  int           `bson:"monitor_interval,omitempty" json:"monitor_interval"`
	MimeTypes        []MimeType    `bson:"mime_types" json:"mime_types"`
	Scenes           []Scene       `bson:"scenes" json:"scenes"`
	Uri              string        `bson:"uri,omitempty" json:"uri"`
	CutIntervalMsecs int           `bson:"cut_interval_msecs,omitempty" json:"cut_interval_msecs,omitempty"`
	StartAt          int64         `bson:"start_at" json:"start_at"`
	EndAt            int64         `bson:"end_at" json:"end_at"`
	Status           SetStatus     `bson:"-" json:"status"`
}

func (s *Set) GenHistory(start, end int64) *SetHistory {
	return &SetHistory{
		SetId:           s.Id,
		Name:            s.Name,
		Type:            s.Type,
		MonitorInterval: s.MonitorInterval,
		MimeTypes:       s.MimeTypes,
		Scenes:          s.Scenes,
		Uri:             s.Uri,
		StartAt:         start,
		EndAt:           end,
		Status:          s.Status,
	}
}

type User struct {
	Id        string `bson:"id" json:"id"`
	Desc      string `bson:"desc" json:"desc"`
	Password  string `bson:"password" json:"-"`
	Roles     []Role `bson:"roles" json:"roles"`
	CreatedAt int64  `bson:"created_at" json:"created_at"`
}

type Entry struct {
	Id               bson.ObjectId       `bson:"_id" json:"id"`
	SetId            string              `bson:"set_id" json:"set_id"`
	Uri              string              `bson:"uri" json:"uri"`
	MimeType         MimeType            `bson:"mime_type" json:"mime_type"`
	Original         *OriginalSuggestion `bson:"original" json:"original"`
	Final            *FinalSuggestion    `bson:"final" json:"final"`
	CoverUri         string              `bson:"cover_uri,omitempty" json:"cover_uri,omitempty"`
	CutIntervalMsecs int                 `bson:"cut_interval_msecs,omitempty" json:"cut_interval_msecs,omitempty"`
	Error            *util.ErrorInfo     `bson:"error" json:"error"`
	CreatedAt        int64               `bson:"created_at" json:"created_at"`
}

func (entry *Entry) GetSceneSuggestions() (res map[Scene]Suggestion, err error) {
	res = make(map[Scene]Suggestion)
	if entry.Original == nil {
		return res, nil
	}

	for _, s := range ValidScenes {
		val, ok := entry.Original.Scenes[s]
		if !ok {
			res[s] = ""
			continue
		}

		var sug CommonSuggestion
		err := util.UnmarshalInterface(val, &sug)
		if err != nil {
			return nil, err
		}
		res[s] = sug.Suggestion
	}

	return res, nil
}

type VideoCut struct {
	Id       bson.ObjectId       `bson:"_id" json:"id"`
	EntryId  string              `bson:"entry_id" json:"-"`
	Uri      string              `bson:"uri" json:"uri"`
	Offset   int64               `bson:"offset" json:"offset"`
	Original *OriginalSuggestion `bson:"original" json:"original"`
}

type OriginalSuggestion struct {
	Suggestion Suggestion            `bson:"suggestion" json:"suggestion"`
	Scenes     map[Scene]interface{} `bson:"scenes" json:"scenes"`
}

type FinalSuggestion struct {
	Suggestion Suggestion           `bson:"suggestion" json:"suggestion"`
	Scenes     map[Scene]Suggestion `bson:"scenes" json:"scenes"`
}

// ===== 图片审核 =====
type ImageCensorReq struct {
	Data struct {
		URI string `json:"uri"`
	} `json:"data"`
	Params struct {
		Scenes []Scene `json:"scenes"`
	} `json:"params"`
}

type ImageCensorResp struct {
	Code    int               `json:"code"`
	Message string            `json:"message"`
	Result  ImageCensorResult `json:"result"`
}

type ImageCensorResult struct {
	Suggestion Suggestion                 `bson:"suggestion" json:"suggestion"` // 审核结论
	Scenes     map[Scene]ImageSceneResult `bson:"scenes" json:"scenes"`
}

type ImageSceneResult struct {
	Suggestion Suggestion `bson:"suggestion" json:"suggestion"` // 审核结论-单场景
	Details    []Detail   `bson:"details" json:"details"`       // 标签明细
}

type Detail struct {
	Suggestion Suggestion    `bson:"suggestion" json:"suggestion"`           // 审核结论-单标签
	Label      string        `bson:"label" json:"label"`                     // 标签
	Group      string        `bson:"group,omitempty" json:"group"`           // 分组
	Score      float32       `bson:"score" json:"score"`                     // 置信度
	Detections []BoundingBox `bson:"detections,omitempty" json:"detections"` // 检测框
}

type BoundingBox struct {
	Pts   [][2]int `json:"pts"`   // 坐标
	Score float32  `json:"score"` //检测框置信度
}

// ===== 视频审核 =====
type VideoCensorReq struct {
	Data struct {
		URI string `json:"uri"`
	} `json:"data"`
	Params struct {
		Scenes   []Scene `json:"scenes"`
		CutParam struct {
			Mode          int `json:"mode"`
			IntervalMsecs int `json:"interval_msecs"`
		} `json:"cut_param"`
		Saver struct {
			Save bool `json:"save"`
		} `json:"saver"`
	} `json:"params"`
}

type VideoCensorResp struct {
	Code    int               `json:"code"`
	Message string            `json:"message"`
	Result  VideoCensorResult `json:"result"`
}

type VideoCensorResult struct {
	Suggestion Suggestion                 `bson:"suggestion" json:"suggestion"` // 审核结论
	Scenes     map[Scene]VideoSceneResult `bson:"scenes" json:"scenes"`
}

type VideoSceneResult struct {
	Suggestion Suggestion  `bson:"suggestion" json:"suggestion"` // 审核结论-单场景
	Cuts       []CutResult `bson:"-" json:"cuts"`                // 帧结果
}

type CutResult struct {
	Suggestion Suggestion `json:"suggestion"`        // 审核结论-单场景-单帧
	Offset     int64      `json:"offset"`            // 帧时间
	Uri        string     `json:"uri,omitempty"`     // 帧存储地址
	Details    []Detail   `json:"details,omitempty"` // 标签明细
}

type CommonResult struct {
	Suggestion Suggestion                 `bson:"suggestion" json:"suggestion"` // 审核结论
	Scenes     map[Scene]CommonSuggestion `bson:"scenes" json:"scenes"`
}
type CommonSuggestion struct {
	Suggestion Suggestion `bson:"suggestion" json:"suggestion"`
}
