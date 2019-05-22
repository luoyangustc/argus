package model

import (
	"qiniu.com/argus/cap/dao"
)

// LabelMode Label Mode
type LabelMode struct {
	Name       string                  `json:"name" bson:"name"`
	LabelTypes []string                `json:"labelTypes" bson:"label_types"`
	Labels     map[string][]LabelTitle `json:"labels" bson:"labels"`
}

func FromLabelInMgo(labelInMgo *dao.LabelInMgo) *LabelMode {

	labels := make(map[string][]LabelTitle)
	for k, v := range labelInMgo.Labels {
		labelTiles := make([]LabelTitle, 0)
		for _, v1 := range v {
			labelTiles = append(labelTiles, LabelTitle{
				Title:    v1.Title,
				Selected: v1.Selected,
			})
		}
		labels[k] = labelTiles
	}
	labelmode := LabelMode{
		Name:       labelInMgo.Name,
		LabelTypes: labelInMgo.LabelTypes,
		Labels:     labels,
	}
	return &labelmode
}

func ToLabelInMgo(labelmode *LabelMode) *dao.LabelInMgo {

	labels := make(map[string][]dao.LabelTitle)
	for k, v := range labelmode.Labels {
		labelTiles := make([]dao.LabelTitle, 0)
		for _, v1 := range v {
			labelTiles = append(labelTiles, dao.LabelTitle{
				Title:    v1.Title,
				Selected: v1.Selected,
			})
		}
		labels[k] = labelTiles
	}
	labelInMgo := dao.LabelInMgo{
		Name:       labelmode.Name,
		LabelTypes: labelmode.LabelTypes,
		Labels:     labels,
	}
	return &labelInMgo
}
