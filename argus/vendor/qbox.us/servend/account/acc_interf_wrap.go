package account

import (
	"net/http"
)

// ---------------------------------------------------------------------------------------
// [[DEPRECATED]]

type OldParser struct {
	Account Interface
}

func (p OldParser) ParseAuth(req *http.Request) (user UserInfo, err error) {
	return GetAuth(p.Account, req)
}

// ---------------------------------------------------------------------------------------
// [[DEPRECATED]]

type OldParserEx struct {
	Account InterfaceEx
}

func (p OldParserEx) ParseAuth(req *http.Request) (user UserInfo, err error) {
	return GetAuthExt(p.Account, req)
}

// ---------------------------------------------------------------------------------------
