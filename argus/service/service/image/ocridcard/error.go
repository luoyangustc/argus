package ocridcard

import (
	"net/http"

	. "qiniu.com/argus/service/service"
)

var (
	ErrIdcardNotFound = func(msg string) ErrorInfo { return NewErrorInfo(http.StatusBadRequest, 802, msg) }
)
