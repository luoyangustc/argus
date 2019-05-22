package sts

import (
	"net/http"

	"github.com/qiniu/http/httputil.v1"
)

//
var (
	ErrNotExist = httputil.NewError(http.StatusNotFound, "file not exist")

	ErrURINotSupport = httputil.NewError(http.StatusBadRequest, "not support uri")
	ErrFetchFailed   = httputil.NewError(http.StatusFailedDependency, "fetch uri failed")
	ErrIO            = httputil.NewError(http.StatusInternalServerError, "io failed")

	ErrOverdue = httputil.NewError(http.StatusInternalServerError, "io overdue")
)
