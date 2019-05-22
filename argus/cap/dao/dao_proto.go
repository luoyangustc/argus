package dao

import (
	"encoding/json"
	"time"

	mgoutil "github.com/qiniu/db/mgoutil.v3"

	"gopkg.in/mgo.v2/bson"
)

type CapMgoConfig struct {
	IdleTimeout  time.Duration  `json:"idle_timeout"`
	MgoPoolLimit int            `json:"mgo_pool_limit"`
	Mgo          mgoutil.Config `json:"mgo"`
}

type JobInMgo struct {
	ID         bson.ObjectId `bson:"_id,omitempty"`
	CreateTime time.Time     `bson:"create_time"`

	JobID     string `bson:"job_id"`
	JobType   string `bson:"job_type"`
	LabelMode string `bson:"label_mode"`
	MimeType  string `bson:"mime_type"` //image || video
	Uid       uint32 `bson:"uid"`       //发起审核请求的用户id
	Status    string `bson:"status"`

	// NotifyURL     string    `bson:"notify_url"` //目前是调用CAP的服务主动来拿结果，所以不需要回调地址
	// Status     string    `bson:"status"` //目前主要以Task的状态来控制，所以删除该字段

	//之前用于计算优先级的，现在按照存量和增量按照一定的比例取task
	// Count      int64     `bson:"count"`
	// Weight     int64     `bson:"weight"`
	// Num        int64     `bson:"num"`
	// Result     string    `bson:"result"`
}

type TaskInMgo struct {
	ID         bson.ObjectId `bson:"_id,omitempty"`
	CreateTime time.Time     `bson:"create_time"`
	UpdateTime time.Time     `bson:"update_time"`

	TaskID string          `bson:"task_id"`
	JobID  string          `bson:"job_id"`
	URI    string          `bson:"uri"`
	Labels json.RawMessage `bson:"labels,omitempty"`
	Status string          `bson:"status"` // todo | doing | done

	AuditorID string          `bson:"auditor_id,omitempty"` //最终完成该标注任务的标注人员Id
	Result    json.RawMessage `bson:"result,omitempty"`     //[]model.LabelInfo
}

// AuditorInMgo
type AuditorInMgo struct {
	ID        bson.ObjectId `bson:"_id,omitempty"`
	CreatedAt time.Time     `bson:"created_at"`
	UpdatedAt time.Time     `bson:"updated_at"`
	Version   int           `bson:"version"`

	AuditorID   string       `bson:"auditor_id"`
	Valid       string       `bson:"valid"`
	CurGroup    string       `bson:"cur_group"`
	AbleGroups  []string     `bson:"able_groups"`
	SandOKNum   int64        `bson:"sand_ok_num"`
	SandAllNum  int64        `bson:"sand_all_num"`
	SandRecords []SandRecord `bson:"sand_records"`
}

// GroupInMgo
type GroupInMgo struct {
	ID        bson.ObjectId `bson:"_id,omitempty"`
	CreatedAt time.Time     `bson:"created_at"`
	UpdatedAt time.Time     `bson:"updated_at"`
	Version   int           `bson:"version"`

	GroupID       string `bson:"group_id"`
	RealTimeLevel string `bson:"real_time_level"`
	LabelModeName string `bson:"label_mode_name"`
	Level         string `bson:"level"`
}

// LabelInMgo
type LabelInMgo struct {
	ID        bson.ObjectId `bson:"_id,omitempty"`
	CreatedAt time.Time     `bson:"created_at"`
	UpdatedAt time.Time     `bson:"updated_at"`
	Version   int           `bson:"version"`

	Name       string                  `bson:"name"`
	LabelTypes []string                `bson:"label_types"`
	Labels     map[string][]LabelTitle `bson:"labels"`
}

type LabelTitle struct {
	Title    string ` bson:"title"`
	Desc     string `bson:"desc"`
	Selected bool   `bson:"selected"`
}

// Sand Model
type SandRecord struct {
	Time    time.Time `bson:"time"`
	TaskID  string    `bson:"task_id"`
	Correct int       `json:"correct"` //0 not correct, 1 correct
}
