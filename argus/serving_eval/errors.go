package eval

import (
	"errors"
	"net/http"

	"github.com/qiniu/http/httputil.v1"
)

//
var (
	ErrRetry = errors.New("need retry")

	ErrBadRequest     = httputil.NewError(http.StatusBadRequest, "bad request")
	ErrOutOfBatchSize = httputil.NewError(http.StatusBadRequest, "out of batch size")
	ErrInvalidImage   = httputil.NewError(http.StatusBadRequest, "invalid image")

	ErrForwardInference = httputil.NewError(http.StatusInternalServerError, "forward inference failed")
)

const (
	CodeErrOutOfBatchSize   = -1
	CodeErrForwardInference = -2
	CodeErrInvalidImage     = -3
)

var (
	EvalErrors = map[int]error{
		CodeErrOutOfBatchSize:   ErrOutOfBatchSize,
		CodeErrForwardInference: ErrForwardInference,
		CodeErrInvalidImage:     ErrInvalidImage,
	}
)

func detectEvalError(code int, message string) error {
	if err, ok := EvalErrors[code]; ok {
		return err
	}
	if code >= 300 {
		return httputil.NewError(code, message)
	}
	return nil
}
