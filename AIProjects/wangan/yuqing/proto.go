package yuqing

import (
	"time"

	"gopkg.in/mgo.v2/bson"
)

type KmqMsg struct {
	Op     string `json:"op"`
	Object struct {
		ID       string `json:"_id"`
		FSize    int64  `json:"fsize"`
		MD5      string `json:"md5"`
		Hash     string `json:"hash"`
		IP       string `json:"ip"`
		Port     string `json:"port"`
		MimeType string `json:"mimeType"`
		PutTime  int64  `json:"putTime"`
		FH       string `json:"fh"`
	} `json:"o"`
}

type Message struct {
	Itbl     string    `json:"itbl" bson:"itbl"`
	Key      string    `json:"key" bson:"key"`
	UID      uint32    `json:"uid" bson:"uid"`
	Bucket   string    `json:"bucket" bson:"bucket"`
	MD5      string    `json:"md5" bson:"md5"`
	URI      string    `json:"uri" bson:"uri"`
	Fsize    int64     `json:"fsize" bson:"fsize"`
	IP       string    `json:"ip" bson:"ip"`
	Port     uint16    `json:"port" bson:"port"`
	MimeType string    `json:"mimeType" bson:"mimeType"`
	FH       string    `json:"fh"`
	PutTime  time.Time `json:"putTime" bson:"putTime"`
}

type BucketEntry struct {
	Itbl    string   `json:"itbl" bson:"itbl"` // bucket index
	Name    string   `json:"name" bson:"name"`
	Uid     uint32   `json:"uid" bson:"uid"`
	Zone    string   `json:"zone" bson:"zone"`
	Domains []string `json:"domains" bson:"domains"`
}
type Metrics struct {
	Total        uint64 `json:"total"`
	TargetVideo  uint64 `json:"video"`
	TargetRegion uint64 `json:"in_region"`
	Inner        uint64 `json:"internal"`
	TargetUser   uint64 `json:"available"`
	Unavailable  uint64 `json:"unavailable"`
}

type OpResultLable struct {
	Name  string  `json:"label" bson:"label"`
	Score float32 `json:"score" bson:"score"`
}

type OpResultCut struct {
	Offset  float32  `json:"offset" bson:"offset"`
	URI     string   `json:"uri" bson:"uri"`
	Classes []string `json:"classes" bson:"classes"`
	Score   float32  `json:"score" bson:"score"`
}

type OpResult struct {
	Labels []OpResultLable `json:"labels" bson:"labels"`
	Cuts   []OpResultCut   `json:"cuts" bson:"cuts"`
}

type VideoResult struct {
	ID           bson.ObjectId       `json:"id" bson:"_id"`
	URI          string              `json:"uri" bson:"uri"`
	UID          uint32              `json:"uid" bson:"uid"`
	Bucket       string              `json:"bucket" bson:"bucket"`
	Key          string              `json:"key" bson:"key"`
	MD5          string              `json:"md5" bson:"md5"`
	Fsize        int64               `json:"fsize" bson:"fsize"`
	UpAddress    string              `json:"up_address" bson:"up_address"`
	MimeType     string              `json:"mimeType" bson:"mimeType"`
	PutTime      time.Time           `json:"putTime" bson:"putTime"`
	Ops          map[string]OpResult `json:"ops" bson:"ops,omitempty"`
	Error        string              `json:"error" bson:"error,omitempty"`
	EvalDuration float64             `json:"eval_duration" bson:"eval_duration"`
}

type MediaType int

const (
	_ MediaType = iota
	MediaTypeImage
	MediaTypeVideo
	MediaTypeAudio
)

type SourceType int

const (
	_ SourceType = iota
	SourceTypeQiniu
	SourceTypeWeibo
	SourceTypeDouyin
	SourceTypeMiaopai
)

type Cut struct {
	Offset float64 `json:"offset"`
	URI    string  `json:"uri"`
	Score  float32 `json:"score"`
}

type Result struct {
	ID           bson.ObjectId       `json:"id" bson:"_id"`
	Type         MediaType           `json:"type" bson:"type"`
	Source       SourceType          `json:"source" bson:"source"`
	URI          string              `json:"uri" bson:"uri"`
	Message      interface{}         `json:"message,omitempty" bson:"message,omitempty"`
	MD5          string              `json:"md5" bson:"md5"`
	Fsize        int64               `json:"fsize" bson:"fsize"`
	ParseTime    time.Time           `json:"parseTime" bson:"parseTime"`
	MimeType     string              `json:"mimeType" bson:"mimeType"`
	Ops          map[string]OpResult `json:"ops,omitempty" bson:"ops,omitempty"`
	Score        float32             `json:"score" bson:"score"`
	Error        string              `json:"error" bson:"error,omitempty"`
	EvalDuration float64             `json:"eval_duration" bson:"eval_duration"`
}

type Job struct {
	URI     string      `json:"uri"`
	Type    MediaType   `json:"type"`
	Source  SourceType  `json:"source"`
	Message interface{} `json:"message,omitempty"`
}

type MiaopaiMessage struct {
	VID         string    `json:"vid" bson:"vid"`
	SMID        string    `json:"smid" bosn:"smid"`
	Description string    `json:"description" bson:"description"`
	CreatedAt   time.Time `json:"created_at" bson:"created_at"`
	ViewsCount  int       `json:"views_count" bson:"views_count"`
	SUID        string    `json:"suid" bson:"suid"`
	Cover       string    `json:"cover" bson:"cover"`
	UserNick    string    `json:"nick" bson:"nick"`
	Birthday    string    `json:"birthday" bson:"birthday"`
}
