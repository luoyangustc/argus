package authstub

import (
	"net/http"
	. "qiniu.com/auth/proto.v1"
)

// ---------------------------------------------------------------------------

type Env struct {
	W   http.ResponseWriter
	Req *http.Request
	UserInfo
}

func (p *Env) OpenEnv(rcvr interface{}, w *http.ResponseWriter, req *http.Request) (err error) {

	auth := req.Header.Get("Authorization")
	user, err := Parse(auth)
	if err != nil {
		return
	}

	p.W = *w
	p.Req = req
	p.UserInfo = user.UserInfo
	return nil
}

func (p *Env) CloseEnv() {
}

// ---------------------------------------------------------------------------

type SudoerEnv struct {
	W   http.ResponseWriter
	Req *http.Request
	SudoerInfo
}

func (p *SudoerEnv) OpenEnv(rcvr interface{}, w *http.ResponseWriter, req *http.Request) (err error) {

	auth := req.Header.Get("Authorization")
	user, err := Parse(auth)
	if err != nil {
		return
	}

	p.W = *w
	p.Req = req
	p.SudoerInfo = user
	return nil
}

func (p *SudoerEnv) CloseEnv() {
}

// ---------------------------------------------------------------------------

