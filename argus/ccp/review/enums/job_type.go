package enums

// JobType define
type JobType string

const (
	JobTypeStream JobType = "STREAM"
	JobTypeBatch  JobType = "BATCH"
)

func (this JobType) IsValid() bool {
	switch this {
	case JobTypeStream, JobTypeBatch:
		return true
	default:
		return false
	}
}
