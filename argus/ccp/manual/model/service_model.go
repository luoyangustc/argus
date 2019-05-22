package model

type BatchEntriesReq struct {
	CmdArgs []string // setID
	Uid     uint32   `json:"uid"`
	Bucket  string   `json:"bucket"`
	Keys    []string `json:"keys"`
}
