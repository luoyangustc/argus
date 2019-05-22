package server

import (
	"net/http"

	"github.com/qiniu/http/httputil.v1"
)

var (
	ErrorImageTooLarge = httputil.NewError(
		http.StatusBadRequest,
		"image is too large, should be in 4999x4999, less than 10MB",
	)

	ErrArgs = httputil.NewError(http.StatusBadRequest, "no enough arguments provided")
)
