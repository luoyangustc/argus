package model

//返回给BCP的结果格式
type NotifyToBCPResponse struct {
	Error  string   `json:"error"`
	Uid    uint32   `json:"uid"`
	Bucket string   `json:"bucket"`
	Keys   []string `json:"keys"`
}
