package enums

type BatchEntryJobStatus uint8

const (
	BatchEntryJobStatusNew BatchEntryJobStatus = iota
	BatchEntryJobStatusProcess
	BatchEntryJobStatusSuccess
	BatchEntryJobStatusFailed
)
