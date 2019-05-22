package model

import (
	ENUMS "qiniu.com/argus/cap/enums"
	"qiniu.com/argus/ccp/manual/dao"
)

type QuerySetsResp struct {
	Result []SetModel `json:"result"`
}

type SaverModel struct {
	UID    uint32  `json:"uid"`
	Bucket string  `json:"bucket"`
	Prefix *string `json:"prefix,omitempty"`
}

type SetModel struct {
	SetId      string        `json:"set_id,omitempty"`
	UID        uint32        `json:"uid"`                   //创建该请求的用户id
	SourceType string        `json:"source_type,omitempty"` // KODO | API
	Type       ENUMS.JobType `json:"type,omitempty"`        // STREAM | BATCH
	Image      *struct {
		IsOn   bool     `json:"is_on"`
		Scenes []string `json:"scenes,omitempty"` // pulp | terror | politician |...
	} `json:"image,omitempty"`
	Video *struct {
		IsOn   bool     `json:"is_on"`
		Scenes []string `json:"scenes,omitempty"` // pulp | terror | politician |...
	} `json:"video,omitempty"`
	NotifyURL string      `json:"notify_url,omitempty"`
	Saver     *SaverModel `json:"saver"`
}

func FromSetInMgo(setInMgo *dao.SetInMgo) *SetModel {
	setModel := SetModel{
		SetId:      setInMgo.SetID,
		UID:        setInMgo.UID,
		SourceType: setInMgo.SourceType,
		Type:       ENUMS.JobType(setInMgo.Type), // stream | batch
		NotifyURL:  setInMgo.NotifyURL,
	}

	setModel.Image = &struct {
		IsOn   bool     `json:"is_on"`
		Scenes []string `json:"scenes,omitempty"` // pulp | terror | politician |...
	}{
		IsOn:   setInMgo.Image.IsOn,
		Scenes: setInMgo.Image.Scenes,
	}

	setModel.Video = &struct {
		IsOn   bool     `json:"is_on"`
		Scenes []string `json:"scenes,omitempty"` // pulp | terror | politician |...
	}{
		IsOn:   setInMgo.Video.IsOn,
		Scenes: setInMgo.Video.Scenes,
	}

	if setInMgo.Saver != nil {
		setModel.Saver = &SaverModel{
			UID:    setInMgo.Saver.UID,
			Bucket: setInMgo.Saver.Bucket,
			Prefix: setInMgo.Saver.Prefix,
		}
	}

	return &setModel
}

func ToSetInMgo(setModel *SetModel) *dao.SetInMgo {
	setInMgo := dao.SetInMgo{
		SetID:      setModel.SetId,
		Type:       string(setModel.Type), // stream | batch
		SourceType: setModel.SourceType,   // KODO | API
		NotifyURL:  setModel.NotifyURL,
		UID:        setModel.UID,
	}

	if setModel.Image != nil {
		setInMgo.Image.IsOn = setModel.Image.IsOn
		setInMgo.Image.Scenes = setModel.Image.Scenes
	}
	if setModel.Video != nil {
		setInMgo.Video.IsOn = setModel.Video.IsOn
		setInMgo.Video.Scenes = setModel.Video.Scenes
	}

	if setModel.Saver != nil {
		setInMgo.Saver = &struct {
			UID    uint32  `bson:"uid"`
			Bucket string  `bson:"bucket"`
			Prefix *string `bson:"prefix"`
		}{
			UID:    setModel.Saver.UID,
			Bucket: setModel.Saver.Bucket,
		}
		if setModel.Saver.Prefix != nil {
			prefix := *setModel.Saver.Prefix
			setInMgo.Saver.Prefix = &prefix
		}
	}

	return &setInMgo
}
