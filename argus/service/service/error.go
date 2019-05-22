package image

import (
	"net/http"
)

var (
// ErrFetchFailed = func(msg string) *httputil.ErrorInfo {
// 	return httputil.NewError(http.StatusFailedDependency, msg)
// }
// ErrorImageType = func(msg string) *httputil.ErrorInfo {
// 	return httputil.NewError(http.StatusUnsupportedMediaType, msg)
// }
// ErrorImageTooLarge = httputil.NewError(
// 	http.StatusBadRequest,
// 	"image is too large, should be in 4999x4999, less than 10MB",
// )
// ErrArgs = httputil.NewError(http.StatusBadRequest, "no enough arguments provided")
)

////////////////////////////////////////////////////////////////////////////////

type DetectErrorer interface {
	DetectError() (int, int, string)
}

type ErrorInfo struct {
	Code    uint32 `json:"code"`
	Message string `json:"message"`
}

func NewErrorInfo(major, minor int, msg string) ErrorInfo {
	// return ErrorInfo{Code: uint32(major)<<12 | uint32(minor), Message: msg}
	return ErrorInfo{Code: uint32(major)*10000 + uint32(minor), Message: msg}
}

func (err ErrorInfo) Error() string {
	return err.Message
}

func (err ErrorInfo) DetectError() (int, int, string) {
	return int(err.Code / 10000), int(err.Code), err.Message
}

var (

	//----------------------------------------------------------------------------//
	// ARGS ERROR
	ErrArgs = func(msg string) ErrorInfo { return NewErrorInfo(http.StatusBadRequest, 100, msg) }

	//----------------------------------------------------------------------------//
	// URI ERROR
	ErrUriNotSupported = func(msg string) ErrorInfo { return NewErrorInfo(http.StatusBadRequest, 201, msg) }
	ErrUriBad          = func(msg string) ErrorInfo { return NewErrorInfo(http.StatusBadRequest, 202, msg) }
	ErrUriFetchFailed  = func(msg string) ErrorInfo { return NewErrorInfo(http.StatusBadRequest, 203, msg) }
	ErrUriFetchTimeout = func(msg string) ErrorInfo { return NewErrorInfo(http.StatusBadRequest, 204, msg) }

	//----------------------------------------------------------------------------//
	// IMAGE ERROR
	ErrImgType       = func(msg string) ErrorInfo { return NewErrorInfo(http.StatusUnsupportedMediaType, 301, msg) }
	ErrImageTooLarge = func(msg string) ErrorInfo { return NewErrorInfo(http.StatusBadRequest, 302, msg) }

	//----------------------------------------------------------------------------//
	// ASYNC JOB ERROR
	ErrAsyncJob = func(msg string) ErrorInfo { return NewErrorInfo(http.StatusInternalServerError, 400, msg) }

	//----------------------------------------------------------------------------//
	//VIDEO ERROR
	ErrVideoType = func(msg string) ErrorInfo { return NewErrorInfo(http.StatusUnsupportedMediaType, 501, msg) }

	//----------------------------------------------------------------------------//
	// INTERNAL ERROR
	ErrInternal = func(msg string) ErrorInfo { return NewErrorInfo(http.StatusInternalServerError, 900, msg) }
)
