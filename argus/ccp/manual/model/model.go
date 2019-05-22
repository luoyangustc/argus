package model

import (
	"encoding/json"

	"qiniu.com/argus/ccp/manual/dao"
)

const (
	IMAGE = "image"
	VIDEO = "video"
)

//=======================================================================================
type EntryModel struct {
	EntryID string `json:"entry_id"`

	Resource json.RawMessage `json:"resource,omitempty"`
	URIGet   string          `json:"uri_get,omitempty"` // 临时的访问地址
	MimeType string          `json:"mimetype"`          // IMAGE / VIDEO / LIVE

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
	Result json.RawMessage `json:"result,omitempty"`
}

func FromEntryInMgo(entryInMgo *dao.EntryInMgo) *EntryModel {
	return &EntryModel{
		EntryID:  entryInMgo.EntryID,
		Resource: entryInMgo.Resource,
		URIGet:   entryInMgo.URIGet,   // 临时的访问地址
		MimeType: entryInMgo.MimeType, // IMAGE / VIDEO / LIVE
		Original: entryInMgo.Original,
		Final:    entryInMgo.Final, // pulp|terror|politician

		// IMAGE: 推理结果
		// VIDEO: 各帧推理结果
		Result: entryInMgo.Result,
	}
}

func ToEntryInMgo(entryModel *EntryModel) *dao.EntryInMgo {
	return &dao.EntryInMgo{
		EntryID:  entryModel.EntryID,
		Resource: entryModel.Resource,
		URIGet:   entryModel.URIGet,   // 临时的访问地址
		MimeType: entryModel.MimeType, // IMAGE / VIDEO / LIVE
		Original: entryModel.Original,
		Final:    entryModel.Final,
		// IMAGE: 推理结果
		// VIDEO: 各帧推理结果
		Result: entryModel.Result,
	}
}
