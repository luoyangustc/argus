package model

import "qiniu.com/argus/ccp/manual/dao"

type BatchEntryModel struct {
	SetId      string `json:"set_id"`
	ImageSetID string `json:"image_set_id"` //cap里面为Image创建的jobid
	VideoSetID string `json:"video_set_id"` //cap里面为Video创建的jobid

	Uid    uint32   `json:"uid"`
	Bucket string   `json:"bucket"`
	Keys   []string `json:"keys"`

	Status  string `json:"status"`
	StartAt int64  `json:"start_at"`
}

func FromBatchEntryInMgo(batchEntryInMgo *dao.BatchEntryInMgo) *BatchEntryModel {
	return &BatchEntryModel{
		SetId:      batchEntryInMgo.SetId,
		ImageSetID: batchEntryInMgo.ImageSetID,
		VideoSetID: batchEntryInMgo.VideoSetID,
		Uid:        batchEntryInMgo.Uid,
		Bucket:     batchEntryInMgo.Bucket,
		Keys:       batchEntryInMgo.Keys,
		Status:     batchEntryInMgo.Status,
		StartAt:    batchEntryInMgo.StartAt,
	}
}
