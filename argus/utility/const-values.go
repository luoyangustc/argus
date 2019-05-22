package utility

import (
	"net/http"

	"github.com/qiniu/http/httputil.v1"
)

var (
	StatusUnkownErrCode = 599
	ErrArgs             = httputil.NewError(http.StatusNotAcceptable, "no enough arguments provided")
	ErrInteServer       = httputil.NewError(StatusUnkownErrCode, "internal error occure in server side")
	ErrNotFound         = httputil.NewError(http.StatusNotFound, "page/path not found")

	DefaultCollSessionPoolLimit = 100
	Base64Header                = "data:application/octet-stream;base64,"
	NoChargeUtype               = uint32(0)
)
