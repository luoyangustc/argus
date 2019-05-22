package enums

type JobStatusType string

const (
	JobBegin  JobStatusType = "begin"
	JobStatus JobStatusType = "closed"
	JobDone   JobStatusType = "done"
)

//============================================================
type JobType string

const (
	REALTIME JobType = "realtime"
	BATCH    JobType = "batch"
)

func (this JobType) IsValid() bool {
	switch this {
	case REALTIME, BATCH:
		return true
	default:
		return false
	}
}

//=============================================================
// type JobType string

// const (
// 	CLASSIFICATION = "classification"
// 	DETECTION      = "detection"
// )

// func (this JobType) IsValid() bool {
// 	switch this {
// 	case CLASSIFICATION, CLASSIFICATION:
// 		return true
// 	default:
// 		return false
// 	}
// }
