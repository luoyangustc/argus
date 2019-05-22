package model

import (
	"context"
	"encoding/json"
	"net/http"
)

const (
	KEY_REQID    = "X-Reqid"
	KEY_LOG      = "X-Log"
	KEY_DURATION = "X-Duration"
	KEY_BATCH    = "X-Batch" // TODO
)

type RequestMessage struct {
	ID       string      `json:"id"`
	Cmd      string      `json:"cmd"`
	Version  *string     `json:"version"`
	Header   http.Header `json:"header"`
	Request  string      `json:"request"`
	Callback string      `json:"callback"`
}

type ResponseMessage struct {
	ID         string      `json:"id"`
	StatusCode int         `json:"status_code"`
	StatusText string      `json:"status_text"`
	Header     http.Header `json:"header"`
	Response   string      `json:"response"`
}

//----------------------------------------------------------------------------//

func NewMessageBody(id string, body []byte) []byte {
	var (
		size  = len([]byte(id))
		_body = make([]byte, 1+size+len(body))
	)
	_body[0] = byte(size)
	copy(_body[1:], []byte(id))
	copy(_body[1+size:], body)
	return _body
}

func ParseMessageBody(body []byte) (string, []byte, error) {
	// TODO check
	var size = int(body[0])
	return string(body[1 : size+1]), body[size+1:], nil
}

//----------------------------------------------------------------------------//

// MarshalRequestMessage ...
func MarshalRequestMessage(
	req TaskReq,
	id, callback string,
	header http.Header,
) ([]byte, error) {

	bs := req.Marshal()
	return json.Marshal(
		RequestMessage{
			ID:       id,
			Cmd:      req.GetCmd(),
			Version:  req.GetVersion(),
			Header:   header,
			Request:  string(bs),
			Callback: callback,
		},
	)
}

// ParseRequestMessage ...
func ParseRequestMessage(bs []byte) (reqM RequestMessage, err error) {
	err = json.Unmarshal(bs, &reqM)
	return
}

// ParseRequest ...
func (msg RequestMessage) ParseRequest(ctx context.Context) (interface{}, error) {
	return UnmarshalTaskRequest(ctx, []byte(msg.Request))
}
