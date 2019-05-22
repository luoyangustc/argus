package gate

import (
	"net/http"

	"github.com/qiniu/http/httputil.v1"
)

//
var (
	ErrNotAcceptable = httputil.NewError(http.StatusNotAcceptable, "not acceptable")
	ErrBadOP         = httputil.NewError(http.StatusBadRequest, "bad op")
	ErrBadURI        = httputil.NewError(http.StatusBadRequest, "bad uri")
	ErrBadDataURI    = httputil.NewError(http.StatusBadRequest, "bad data uri")

	ErrTimeout = httputil.NewError(http.StatusInternalServerError, "timeout")
)
