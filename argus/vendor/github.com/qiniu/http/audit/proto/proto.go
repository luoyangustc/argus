package proto

// ----------------------------------------------------------

type M map[string]interface{}

type Request struct {
	Mod       string `json:"mod"`
	Method    string `json:"method"`
	Path      string `json:"path"`
	Header    M      `json:"header"`
	Params    M      `json:"params"`
	StartTime int64  `json:"start"`
}

type Event interface {
	OnStartReq(req *Request) (id interface{}, err error)
	OnEndReq(id interface{})
}

// ----------------------------------------------------------
