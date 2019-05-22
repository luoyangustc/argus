package proto

import (
	"encoding/json"
	"errors"
	"time"
)

const (
	RULE_STATUS_ON  = "ON"
	RULE_STATUS_OFF = "OFF"

	TYPE_STREAM = "STREAM"
	TYPE_BATCH  = "BATCH"

	SRC_KODO = "KODO"
	SRC_API  = "API"
)

// Rule model
type Rule struct {
	RuleID    string `json:"rule_id"` // RuleID
	UID       uint32 `json:"uid"`
	Utype     uint32 `json:"utype,omitempty"`
	CreateSec int64  `json:"create_sec"` // timestamp for Rule Create
	EndSec    int64  `json:"end_sec"`    // timestamp for Rule End
	Status    string `json:"status"`     // ON | OFF

	Type       string          `json:"type"`             // STREAM | BATCH
	SourceType string          `json:"source_type"`      // KODO | API | FUSION
	SourceID   string          `json:"source_id"`        // Src表的ID
	Source     json.RawMessage `json:"source,omitempty"` // like KodoSource
	Action     json.RawMessage `json:"action,omitempty"` // like KodoAction

	NotifyURL *string  `json:"notify_url,omitempty"`
	Saver     struct { // for tmp save
		IsOn   bool    `json:"is_on"`
		UID    uint32  `json:"uid"`
		Bucket string  `json:"bucket"`
		Prefix *string `json:"prefix,omitempty"`
	} `json:"saver"`

	Image struct {
		IsOn   bool     `json:"is_on"`
		Scenes []string `json:"scenes,omitempty"` // pulp | terror | ...
	} `json:"image,omitempty"`
	Video struct {
		IsOn   bool     `json:"is_on"`
		Scenes []string `json:"scenes,omitempty"` // pulp | terror | ...
	} `json:"video,omitempty"`

	Automatic struct {
		IsOn  bool    `json:"is_on"`
		JobID *string `json:"job_id,omitempty"`
		Image struct {
			SceneParams map[string]json.RawMessage `json:"scene_params,omitempty"`
		} `json:"image,omitempty"`
		Video struct {
			Params      map[string]json.RawMessage `json:"params,omitempty"`
			SceneParams map[string]json.RawMessage `json:"scene_params,omitempty"`
		} `json:"video,omitempty"`
	} `json:"automatic,omitempty"`
	Manual struct {
		IsOn  bool    `json:"is_on"`
		JobID *string `json:"job_id,omitempty"`
		Image struct {
			SceneParams map[string]json.RawMessage `json:"scene_params,omitempty"`
		} `json:"image,omitempty"`
		Video struct {
			SceneParams map[string]json.RawMessage `json:"scene_params,omitempty"`
		} `json:"video,omitempty"`
	} `json:"manual,omitempty"`
	Review struct {
		IsOn bool    `json:"is_on"`
		ID   *string `json:"id,omitempty"` // 生成任务时，暂时不用指定该字段
	} `json:"review"`
}

//================================================================

type RuleInMgo struct {
	RuleID     string    `json:"rule_id" bson:"rule_id"`
	UID        uint32    `json:"uid" bson:"uid"`
	Utype      uint32    `json:"utype,omitempty" bson:"utype,omitempty"`
	CreateTime time.Time `json:"create_time" bson:"create_time"`
	EndTime    time.Time `json:"end_time" bson:"end_time"`
	Status     string    `json:"status" bson:"status"` // ON | OFF

	Type       string      `json:"type" bson:"type"`               // STREAM | BATCH
	SourceType string      `json:"source_type" bson:"source_type"` // KODO | API | FUSION
	SourceID   string      `json:"source_id" bson:"source_id"`     // Src表的ID
	Action     interface{} `json:"action" bson:"action"`           // like KodoAction

	NotifyURL *string `json:"notify_url,omitempty" bson:"notify_url,omitempty"`
	Saver     struct {
		IsOn   bool    `json:"is_on" bson:"is_on"`
		UID    uint32  `json:"uid" bson:"uid"`
		Bucket string  `json:"bucket" bson:"bucket"`
		Prefix *string `json:"prefix,omitempty" bson:"prefix,omitempty"`
	} `json:"saver" bson:"saver"`

	Image struct {
		IsOn   bool     `json:"is_on" bson:"is_on"`
		Scenes []string `json:"scenes,omitempty" bson:"scenes,omitempty"`
	} `json:"image,omitempty" bson:"image,omitempty"`
	Video struct {
		IsOn   bool     `json:"is_on" bson:"is_on"`
		Scenes []string `json:"scenes,omitempty" bson:"scenes,omitempty"`
	} `json:"video,omitempty" bson:"video,omitempty"`

	Automatic struct {
		IsOn  bool    `json:"is_on" bson:"is_on"`
		JobID *string `json:"job_id,omitempty" bson:"job_id,omitempty"`
		Image struct {
			SceneParams map[string]interface{} `json:"scene_params,omitempty" bson:"scene_params,omitempty"`
		} `json:"image,omitempty" bson:"image,omitempty"`
		Video struct {
			Params      map[string]interface{} `json:"params,omitempty" bson:"params,omitempty"`
			SceneParams map[string]interface{} `json:"scene_params,omitempty" bson:"scene_params,omitempty"`
		} `json:"video,omitempty" bson:"video,omitempty"`
	} `json:"automatic,omitempty" bson:"automatic,omitempty"`

	Manual struct {
		IsOn  bool    `json:"is_on" bson:"is_on"`
		JobID *string `json:"job_id,omitempty" bson:"job_id,omitempty"`
		Image struct {
			SceneParams map[string]interface{} `json:"scene_params,omitempty" bson:"scene_params,omitempty"`
		} `json:"image,omitempty" bson:"image,omitempty"`
		Video struct {
			SceneParams map[string]interface{} `json:"scene_params,omitempty" bson:"scene_params,omitempty"`
		} `json:"video,omitempty" bson:"video,omitempty"`
	} `json:"manual,omitempty" bson:"manual,omitempty"`

	Review struct {
		IsOn bool    `json:"is_on" bson:"is_on"`
		ID   *string `json:"id,omitempty" bson:"id,omitempty"` // 生成任务时，暂时不用指定该字段
	} `json:"review" bson:"review"`
}

//================================================================

func (ruleInMgo *RuleInMgo) FromRule(rule *Rule) error {

	if rule == nil {
		return errors.New("convert failed, invalid params")
	}

	ruleInMgo.RuleID = rule.RuleID
	ruleInMgo.UID = rule.UID
	ruleInMgo.Utype = rule.Utype
	ruleInMgo.CreateTime = time.Unix(rule.CreateSec, 0)
	ruleInMgo.EndTime = time.Unix(rule.EndSec, 0)
	ruleInMgo.Status = rule.Status
	ruleInMgo.Type = rule.Type
	ruleInMgo.SourceType = rule.SourceType
	ruleInMgo.SourceID = rule.SourceID
	ruleInMgo.NotifyURL = rule.NotifyURL

	if len(rule.Action) > 0 {
		err := json.Unmarshal(rule.Action, &ruleInMgo.Action)
		if err != nil {
			return err
		}
	}

	ruleInMgo.Saver.IsOn = rule.Saver.IsOn
	if ruleInMgo.Saver.IsOn {
		ruleInMgo.Saver.UID = rule.Saver.UID
		ruleInMgo.Saver.Bucket = rule.Saver.Bucket
		ruleInMgo.Saver.Prefix = rule.Saver.Prefix
	}

	ruleInMgo.Image.IsOn = rule.Image.IsOn
	if ruleInMgo.Image.IsOn {
		ruleInMgo.Image.Scenes = rule.Image.Scenes
	}

	ruleInMgo.Video.IsOn = rule.Video.IsOn
	if ruleInMgo.Video.IsOn {
		ruleInMgo.Video.Scenes = rule.Video.Scenes
	}

	ruleInMgo.Automatic.IsOn = rule.Automatic.IsOn
	if ruleInMgo.Automatic.IsOn {
		ruleInMgo.Automatic.JobID = rule.Automatic.JobID
		ruleInMgo.Automatic.Image.SceneParams = make(map[string]interface{})
		for k, v := range rule.Automatic.Image.SceneParams {
			var tmp interface{}
			err := json.Unmarshal(v, &tmp)
			if err != nil {
				return err
			}
			ruleInMgo.Automatic.Image.SceneParams[k] = tmp
		}

		ruleInMgo.Automatic.Video.SceneParams = make(map[string]interface{})
		for k, v := range rule.Automatic.Video.SceneParams {
			var tmp interface{}
			err := json.Unmarshal(v, &tmp)
			if err != nil {
				return err
			}
			ruleInMgo.Automatic.Video.SceneParams[k] = tmp
		}

		ruleInMgo.Automatic.Video.Params = make(map[string]interface{})
		for k, v := range rule.Automatic.Video.Params {
			var tmp interface{}
			err := json.Unmarshal(v, &tmp)
			if err != nil {
				return err
			}
			ruleInMgo.Automatic.Video.Params[k] = tmp
		}
	}

	ruleInMgo.Manual.IsOn = rule.Manual.IsOn
	if ruleInMgo.Manual.IsOn {
		ruleInMgo.Manual.JobID = rule.Manual.JobID
		ruleInMgo.Manual.Image.SceneParams = make(map[string]interface{})
		for k, v := range rule.Manual.Image.SceneParams {
			var tmp interface{}
			err := json.Unmarshal(v, &tmp)
			if err != nil {
				return err
			}
			ruleInMgo.Manual.Image.SceneParams[k] = tmp
		}

		ruleInMgo.Manual.Video.SceneParams = make(map[string]interface{})
		for k, v := range rule.Manual.Video.SceneParams {
			var tmp interface{}
			err := json.Unmarshal(v, &tmp)
			if err != nil {
				return err
			}
			ruleInMgo.Manual.Video.SceneParams[k] = tmp
		}
	}

	ruleInMgo.Review.IsOn = rule.Review.IsOn
	if ruleInMgo.Review.IsOn {
		ruleInMgo.Review.ID = rule.Review.ID
	}

	return nil
}

func (ruleInMgo *RuleInMgo) ToRule(src json.RawMessage, rule *Rule) error {

	if rule == nil {
		return errors.New("convert failed, invalid params")
	}

	rule.RuleID = ruleInMgo.RuleID
	rule.UID = ruleInMgo.UID
	rule.Utype = ruleInMgo.Utype
	rule.CreateSec = ruleInMgo.CreateTime.Unix()
	rule.EndSec = ruleInMgo.EndTime.Unix()
	rule.Status = ruleInMgo.Status
	rule.Type = ruleInMgo.Type
	rule.SourceType = ruleInMgo.SourceType
	rule.SourceID = ruleInMgo.SourceID
	rule.Source = src
	rule.NotifyURL = ruleInMgo.NotifyURL

	var err error
	rule.Action, err = json.Marshal(ruleInMgo.Action)
	if err != nil {
		return err
	}

	rule.Saver.IsOn = ruleInMgo.Saver.IsOn
	if rule.Saver.IsOn {
		rule.Saver.UID = ruleInMgo.Saver.UID
		rule.Saver.Bucket = ruleInMgo.Saver.Bucket
		rule.Saver.Prefix = ruleInMgo.Saver.Prefix
	}

	rule.Image.IsOn = ruleInMgo.Image.IsOn
	if rule.Image.IsOn {
		rule.Image.Scenes = ruleInMgo.Image.Scenes
	}

	rule.Video.IsOn = ruleInMgo.Video.IsOn
	if rule.Video.IsOn {
		rule.Video.Scenes = ruleInMgo.Video.Scenes
	}

	rule.Automatic.IsOn = ruleInMgo.Automatic.IsOn
	if rule.Automatic.IsOn {
		rule.Automatic.JobID = ruleInMgo.Automatic.JobID
		rule.Automatic.Image.SceneParams = make(map[string]json.RawMessage)
		for k, v := range ruleInMgo.Automatic.Image.SceneParams {
			tmp, err := json.Marshal(v)
			if err != nil {
				return err
			}
			rule.Automatic.Image.SceneParams[k] = tmp
		}

		rule.Automatic.Video.SceneParams = make(map[string]json.RawMessage)
		for k, v := range ruleInMgo.Automatic.Video.SceneParams {
			tmp, err := json.Marshal(v)
			if err != nil {
				return err
			}
			rule.Automatic.Video.SceneParams[k] = tmp
		}

		rule.Automatic.Video.Params = make(map[string]json.RawMessage)
		for k, v := range ruleInMgo.Automatic.Video.Params {
			tmp, err := json.Marshal(v)
			if err != nil {
				return err
			}
			rule.Automatic.Video.Params[k] = tmp
		}
	}

	rule.Manual.IsOn = ruleInMgo.Manual.IsOn
	if rule.Manual.IsOn {
		rule.Manual.JobID = ruleInMgo.Manual.JobID
		rule.Manual.Image.SceneParams = make(map[string]json.RawMessage)
		for k, v := range ruleInMgo.Manual.Image.SceneParams {
			tmp, err := json.Marshal(v)
			if err != nil {
				return err
			}
			rule.Manual.Image.SceneParams[k] = tmp
		}

		rule.Manual.Video.SceneParams = make(map[string]json.RawMessage)
		for k, v := range ruleInMgo.Manual.Video.SceneParams {
			tmp, err := json.Marshal(v)
			if err != nil {
				return err
			}
			rule.Manual.Video.SceneParams[k] = tmp
		}
	}

	rule.Review.IsOn = ruleInMgo.Review.IsOn
	if rule.Review.IsOn {
		rule.Review.ID = ruleInMgo.Review.ID
	}

	return nil
}
