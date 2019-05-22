package image

import (
	"context"
	"encoding/json"
)

type Suggestion string

const (
	PASS   Suggestion = "pass"
	BLOCK  Suggestion = "block"
	REVIEW Suggestion = "review"
)

const (
	CENSOR_BY_LABEL = "label" // 根据label字段来判定
	CENSOR_BY_GROUP = "group" // 根据group字段来判定
)

func (suggestion Suggestion) Update(newV Suggestion) Suggestion {
	if suggestion == BLOCK {
		return suggestion
	}
	if newV == BLOCK {
		return newV
	}
	if suggestion == REVIEW {
		return suggestion
	}
	if newV == REVIEW {
		return newV
	}
	return newV
}

//----------------------------------------------------------------------------//

type ImageCensorReq struct {
	Data struct {
		IMG Image
	} `json:"data"`
	Params json.RawMessage `json:"params,omitempty"`
}

type SceneResult struct {
	Suggestion Suggestion `json:"suggestion"`        // 审核结论-单场景
	Details    []Detail   `json:"details,omitempty"` // 标签明细
}

type Detail struct {
	Suggestion Suggestion    `json:"suggestion"`           // 审核结论-单标签
	Label      string        `json:"label"`                // 标签
	Group      string        `json:"group,omitempty"`      // 分组
	Score      float32       `json:"score"`                // 置信度
	Comments   string        `json:"comments,omitempty"`   // 提示描述
	Detections []BoundingBox `json:"detections,omitempty"` // 检测框
}

type BoundingBox struct {
	Pts      [][2]int `json:"pts"`                // 坐标
	Score    float32  `json:"score"`              //检测框置信度
	Comments string   `json:"comments,omitempty"` // 提示描述
}

type SugConfig struct {
	CensorBy string `json:"censor_by" bson:"censor_by"` // 根据哪个维度判定
	// 每个标签都有对应的配置，可通过改变其各threshold来控制输不输出该标签，并且可控制其审核suggestion
	Rules map[string]RuleConfig `json:"rules,omitempty" bson:"rules,omitempty"`
}

type RuleConfig struct {
	AbandonThreshold float32    `json:"abandon_threshold" bson:"abandon_threshold"` // 小于该置信度则该标签不输出
	SureThreshold    float32    `json:"sure_threshold" bson:"sure_threshold"`       // 大于该置信度则为确信，小于为不确信
	SureSuggestion   Suggestion `json:"sure_suggestion" bson:"sure_suggestion"`     // 确信时的suggestin
	UnsureSuggestion Suggestion `json:"unsure_suggestion" bson:"unsure_suggestion"` // 不确信时的suggestin
}

//--------------------------------------------------------------------------------//

func V(v interface{}) func() interface{} { return func() interface{} { return v } }

func Co(c bool, f1, f2 func() interface{}) interface{} {
	if c {
		return f1()
	} else {
		return f2()
	}
}

// 合并结果
func MergeDetails(ctx context.Context, sugcfg *SugConfig,
	origDetails []Detail, detail ...Detail) *SceneResult {

	var (
		suggestion = PASS
		rawDetails = make([]Detail, 0, len(origDetails)+len(detail))
		details    = make([]Detail, 0, len(origDetails)+len(detail))
	)

	rawDetails = append(rawDetails, origDetails...)
	rawDetails = append(rawDetails, detail...)

	// 合并相同label
	for _, d := range rawDetails {
		var (
			index int
			exist bool
		)
		for i, d2 := range details {
			if d2.Label == d.Label {
				exist = true
				index = i
				break
			}
		}

		if exist {
			details[index].Suggestion = details[index].Suggestion.Update(d.Suggestion)
			details[index].Detections = append(details[index].Detections, d.Detections...)
			if details[index].Score < d.Score {
				details[index].Score = d.Score
				details[index].Comments = d.Comments
			}
		} else {
			details = append(details, d)
		}
	}

	for _, d := range details {
		if d.Label == "" {
			continue
		}

		suggestion = suggestion.Update(d.Suggestion)
	}

	sceneResult := SceneResult{
		Suggestion: suggestion,
		Details:    details,
	}

	return &sceneResult
}

// 通过config对单标签判定结果，若返回值为false则该标签应抛弃
func (detail *Detail) SetSuggestion(sugcfg *SugConfig) bool {
	var tag string
	if sugcfg.CensorBy == CENSOR_BY_GROUP {
		// 按照分组判定
		tag = detail.Group
	} else {
		// 默认按标签判定
		tag = detail.Label
	}

	rule, ok := sugcfg.Rules[tag]

	if ok {
		if detail.Score < rule.AbandonThreshold {
			// 小于抛弃置信度则该标签不输出
			return false
		}

		var sugg Suggestion
		if detail.Score < rule.SureThreshold {
			sugg = rule.UnsureSuggestion
		} else {
			sugg = rule.SureSuggestion
		}

		switch sugg {
		case PASS, REVIEW, BLOCK:
			detail.Suggestion = sugg
		default:
			detail.Suggestion = PASS
		}

		return true
	}

	// 默认正常
	detail.Suggestion = PASS
	return true
}
