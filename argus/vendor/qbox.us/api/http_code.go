package api

import (
	"qbox.us/errors"

	"github.com/qiniu/http/httputil.v1"
	"github.com/qiniu/rpc.v1"
)

// --------------------------------------------------------------------

var g_httpcodes = map[error]int{
	EInvalidArgs:          InvalidArgs,
	EBadToken:             BadToken,
	EBadOAuthRequest:      BadOAuthRequest,
	EBadRequestMethod:     BadRequestMethod,
	EDataVerificationFail: DataVerificationFail,
	ETooManyRequests:      TooManyRequests,
	EProcessPanic:         ProcessPanic,
	EVersionTooOld:        VersionTooOld,
	EFunctionFail:         FunctionFail,
	ENotImpl:              NotImpl,
	EPartialOK:            PartialOK,
}

func HttpCode(err error) int {
	err = errors.Err(err)
	if code, ok := g_httpcodes[err]; ok {
		return code
	}
	if e, ok := err.(Errno); ok {
		return int(e)
	}
	if e, ok := err.(*rpc.ErrorInfo); ok {
		return e.Code
	}
	if e, ok := err.(*httputil.ErrorInfo); ok {
		return e.Code
	}
	return FunctionFail
}

// --------------------------------------------------------------------

func RegisterError(code int, err string) error {
	e := errors.Register(err)
	g_errors[code] = e
	g_httpcodes[e] = code
	return e
}

// --------------------------------------------------------------------
