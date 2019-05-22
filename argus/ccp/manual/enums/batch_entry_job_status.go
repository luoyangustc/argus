package enums

//type BatchEntryJobStatus uint8

const (
	BatchEntryJobStatusNew     = "new"
	BatchEntryJobStatusProcess = "process" //开始读取该job里面的机审结果文件
	BatchEntryJobStatusEnd     = "end"     //job里面的机审结果文件处理完成并开始等待人审结束
	BatchEntryJobStatusSuccess = "success"
	BatchEntryJobStatusFailed  = "fail"
)
