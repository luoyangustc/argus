package dao

import (
	"encoding/json"
	"time"

	mgoutil "github.com/qiniu/db/mgoutil.v3"

	"gopkg.in/mgo.v2/bson"
)

type CcpCapMgoConfig struct {
	IdleTimeout  time.Duration  `json:"idle_timeout"`
	MgoPoolLimit int            `json:"mgo_pool_limit"`
	Mgo          mgoutil.Config `json:"mgo"`
}

//==================================================================
type SetInMgo struct {
	ID        bson.ObjectId `bson:"_id,omitempty"`
	SetID     string        `bson:"set_id"`
	CreatedAt time.Time     `bson:"created_at"`
	UpdatedAt time.Time     `bson:"updated_at"`

	UID        uint32 `bson:"uid"`
	SourceType string // KODO | API
	Type       string `bson:"type"` // STREAM | BATCH
	NotifyURL  string `bson:"notify_url"`
	Error      string `bson:"error"`

	Image struct {
		IsOn   bool     `bson:"is_on"`
		Scenes []string `bson:"scenes"` // pulp | terror | politician |...
	} `bson:"image"`
	Video struct {
		IsOn   bool     `bson:"is_on"`
		Scenes []string `bson:"scenes"` // pulp | terror | politician |...
	} `bson:"video"`

	//人审结果的存储信息
	Saver *struct {
		UID    uint32  `bson:"uid"`
		Bucket string  `bson:"bucket"`
		Prefix *string `bson:"prefix"`
	}
	ResultFiles []string `bson:"result_files"` //人审处理完后存在bucket里面的keys
}

//======================================================================================
//存量任务
type BatchEntryInMgo struct {
	ID         bson.ObjectId `bson:"_id"`
	SetId      string        `bson:"set_id"`
	ImageSetID string        `bson:"image_set_id"` //cap里面为Image创建的jobid
	VideoSetID string        `bson:"video_set_id"` //cap里面为Video创建的jobid
	CreatedAt  time.Time     `bson:"created_at"`
	UpdatedAt  time.Time     `bson:"updated_at"`

	//要人审的文件
	Uid    uint32   `bson:"uid"`
	Bucket string   `bson:"bucket"`
	Keys   []string `bson:"keys"`

	Status  string `bson:"status"`
	StartAt int64  `bson:"start_at"`
}

//==========================================================================================//
//增量任务 TODO
type EntryInMgo struct { //Entry信息
	ID        bson.ObjectId `bson:"_id,omitempty"`
	EntryID   string        `bson:"entry_id"`
	SetID     string        `bson:"set_id"`
	CreatedAt time.Time     `bson:"created_at"`
	UpdatedAt time.Time     `bson:"updated_at"`

	Resource json.RawMessage `bson:"resource,omitempty"`
	URIGet   string          `bson:"uri_get,omitempty"` // 临时的访问地址
	MimeType string          `bson:"mimetype"`          // IMAGE / VIDEO / LIVE

	Original struct {
		Source     string              `json:"source"`
		Suggestion byte                `json:"suggestion"` // PASS: 0, BLOCK: 1, REVIEW: 2
		Scenes     map[string]struct { // pulp|terror|politician
			Suggestion byte    `json:"Suggestion"`
			Score      float32 `json:"socre,omitempty"`
		} `json:"scenes"`
	}
	Final struct {
		Suggestion byte            `json:"Suggestion"` // PASS: 0, BLOCK: 1, REVIEW: 2
		Scenes     map[string]byte `json:"scenes"`     // pulp|terror|politician
	}

	// IMAGE: 推理结果
	// VIDEO: 各帧推理结果
	Result json.RawMessage `bson:"result,omitempty"`
}
