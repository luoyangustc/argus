package model

//
const (
	BatchLevel    string = "batch"
	RealTimeLevel string = "realtime"
)

// Auditor ...
type Auditor struct {
	ID         string   `json:"id" bson:"id"`
	Valid      string   `json:"valid" bson:"valid"`
	CurGroup   string   `json:"curGroup" bson:"cur_group"`
	AbleGroups []string `json:"ableGroups" bson:"able_groups"`
	SandOKNum  int64    `json:"sandOkNum" bson:"sand_ok_num"`
	SandAllNum int64    `json:"sandAllNum" bson:"sand_all_num"`
}

// AuditorGroup ...
type AuditorGroup struct {
	GroupID       string `json:"groupId" bson:"group_id"`
	Mode          string `json:"mode" bson:"mode"`
	RealTimeLevel string `json:"realtimeLevel" bson:"real_time_level"`
	Level         string `json:"level" bson:"level"`
}

type AuditorConfig struct {
	IntervalSecs        int64 `json:"interval_secs"`
	SingleTimeoutSecs   int64 `json:"single_timeout_secs"`
	MaxTasksNum         int   `json:"max_tasks_num"`
	PackSize            int   `json:"pack_size"`
	NoSandLimitint      int   `json:"no_sand_limit"`
	SandPercentage      int   `json:"sand_percentage"`
	RecordReserveSecond int64 `json:"record_reserve_second"` //sand record 保留时间
}

//=========================

//	service_audit
////////////////////////////////////////////////////////////////
type AuditorModel struct {
	AuditorID  string `json:"auditor_id"`
	Valid      string `json:"valid"`
	CurGroupID string `json:"cur_group"` // group_id

	//Label表中的信息
	LabelModeName string                  `json:"label_mode_name"` // mode_pulp, mode_pulp_terror, ...
	LabelTypes    []string                `json:"label_types"`     // classify.pulp, classify.terror, classify.politician
	Labels        map[string][]LabelTitle `json:"label"`           // pulp, terror, politician的子标签
}
