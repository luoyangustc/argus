package face

import (
	"net/http"

	. "qiniu.com/argus/service/service"
)

var (
	ErrFaceNotFound = func(msg string) ErrorInfo { return NewErrorInfo(http.StatusBadRequest, 601, msg) }
)
