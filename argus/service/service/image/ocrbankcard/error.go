package ocrbankcard

import (
	"net/http"

	. "qiniu.com/argus/service/service"
)

var (
	ErrBankcardNotFound = func(msg string) ErrorInfo { return NewErrorInfo(http.StatusBadRequest, 803, msg) }
)
