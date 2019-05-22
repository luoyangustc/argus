package api

import (
	"qbox.us/errors"
	"strconv"
)

// --------------------------------------------------------------------
// HTTP服务端错误

const (
	OK        = 200
	PartialOK = 298 // Partial OK

	InvalidArgs          = 400 // Bad input parameter. Error message should indicate which one and why.
	BadToken             = 401 // Token 授权错误（Access Token 超时，用户修改了密码，或输入的密码错）
	BadOAuthRequest      = 403 // Bad OAuth request (wrong consumer token, bad nonce, expired timestamp, …).
	BadRequestMethod     = 405 // Request method not expected (generally should be GET or POST).
	DataVerificationFail = 406

	TooManyRequests = 503 // 请求过频繁
	NotImpl         = 596 // 未实现
	ProcessPanic    = 597 // 请求处理发生异常
	VersionTooOld   = 598 // 客户端版本过老，支持的协议已经被废除
	FunctionFail    = 599 // 请求未完成
)

var (
	EInvalidArgs          = errors.EINVAL
	EPartialOK            = errors.Register("Partial OK")
	EBadToken             = errors.Register("bad token")
	EBadOAuthRequest      = errors.Register("bad oauth request")
	EBadRequestMethod     = errors.Register("bad request method")
	EDataVerificationFail = errors.Register("data verification fail")
	ETooManyRequests      = errors.Register("too many requests")
	EProcessPanic         = errors.Register("process panic")
	EVersionTooOld        = errors.Register("version too old")
	EFunctionFail         = errors.Register("function fail")
	ENotImpl              = errors.Register("not impl")
)

// --------------------------------------------------------------------
// 客户端错误

const (
	NetworkError       = 996 // 网络错误(非TimeoutError)。
	TimeoutError       = 997 // 请求超时。
	UnexpectedResponse = 998 // 非预期的输出。see api.UnexpectedResponse
	InternalError      = 999 // 内部错误。see api.InternalError
)

var (
	ENetworkError       = errors.Register("network error")
	ETimeoutError       = errors.Register("timeout")
	EUnexpectedResponse = errors.Register("unexpected response")
	EInternalError      = errors.Register("internal error")
)

// --------------------------------------------------------------------

var g_errors = map[int]error{

	InvalidArgs:          EInvalidArgs,
	BadToken:             EBadToken,
	BadOAuthRequest:      EBadOAuthRequest,
	BadRequestMethod:     EBadRequestMethod,
	DataVerificationFail: EDataVerificationFail,
	TooManyRequests:      ETooManyRequests,
	ProcessPanic:         EProcessPanic,
	VersionTooOld:        EVersionTooOld,
	FunctionFail:         EFunctionFail,
	NotImpl:              ENotImpl,
	PartialOK:            EPartialOK,

	NetworkError:       ENetworkError,
	TimeoutError:       ETimeoutError,
	UnexpectedResponse: EUnexpectedResponse,
	InternalError:      EInternalError,
}

// --------------------------------------------------------------------

type Errno int

func (r Errno) Error() string {
	return "E" + strconv.Itoa(int(r))
}

func NewError(code int) error {
	if e, ok := g_errors[code]; ok {
		return e
	}
	return Errno(code)
}

// --------------------------------------------------------------------
