package gate

import "qiniu.com/argus/atserving/model"

// TODO: 这里的数据结构和 ava/platform/src/qiniu.com/argus/atflow/logexporter/proto/log.go 重用

// AI产品日志
type aiprdLogMsg struct {
	Module            string                  `json:"module,omitempty"`
	Reqid             string                  `json:"reqid,omitempty"`
	UID               uint32                  `json:"uid,omitempty"`
	Cmd               string                  `json:"cmd,omitempty"`
	Version           string                  `json:"version,omitempty"`
	EvalRequest       *model.EvalRequest      `json:"eval_request,omitempty"`
	GroupEvalRequest  *model.GroupEvalRequest `json:"group_eval_request,omitempty"`
	EvalResponse      *model.EvalResponse     `json:"eval_response,omitempty"`
	EvalBatchResponse interface{}             `json:"eval_batch_response,omitempty"`
}

type aiprdLog struct {
	AiprdLog aiprdLogMsg `json:"aiprdlog"`
	URIData  [][]byte    `json:"uri_data"`
	uri      []string
}

func newAiprdlog(name string) *aiprdLog {
	return &aiprdLog{AiprdLog: aiprdLogMsg{Module: "AVA-PRODUCT-LOG-" + name}}
}

func (a *aiprdLog) addBase(reqid string, uid uint32, cmd string, version *string) {
	a.AiprdLog.Reqid = reqid
	a.AiprdLog.UID = uid
	a.AiprdLog.Cmd = cmd
	if version != nil {
		a.AiprdLog.Version = *version
	}
}

func (a *aiprdLog) addEvalRequest(r *model.EvalRequest) {
	a.AiprdLog.EvalRequest = r
}

func (a *aiprdLog) addGroupEvalRequest(r *model.GroupEvalRequest) {
	a.AiprdLog.GroupEvalRequest = r
}
func (a *aiprdLog) addEvalResponse(r *model.EvalResponse) {
	a.AiprdLog.EvalResponse = r
}

func (a *aiprdLog) addEvalBatchResponse(r interface{}) {
	a.AiprdLog.EvalBatchResponse = r
}

// func (a *aiprdLog) marshal() (buf []byte) {
// 	buf, _ = json.Marshal(a.AiprdLog)
// 	return
// }

func (a *aiprdLog) addUri(uri string) {
	a.uri = append(a.uri, uri)
}
