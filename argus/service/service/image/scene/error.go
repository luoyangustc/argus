package scene

import (
	"net/http"

	. "qiniu.com/argus/service/service"
)

var (
	ErrTextNotFound = func(msg string) ErrorInfo { return NewErrorInfo(http.StatusBadRequest, 801, msg) }
)
