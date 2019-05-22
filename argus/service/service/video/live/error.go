package live

import (
	"net/http"

	httputil "github.com/qiniu/http/httputil.v1"

	. "qiniu.com/argus/service/service"
)

func formatError(err error) error {
	if err == nil {
		return nil
	}
	if _, ok := err.(DetectErrorer); ok {
		return err
	}
	switch err {
	}
	switch err2 := err.(type) {
	case *httputil.ErrorInfo:
		switch err2.Code {
		case http.StatusBadRequest:
			return ErrArgs(err.Error())
		default:
			return ErrInternal(err.Error())
		}
	default:
		return ErrInternal(err.Error())
	}
}
