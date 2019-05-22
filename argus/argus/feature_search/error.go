// +build cublas

package feature_search

import (
	"errors"
	"net/http"

	"github.com/qiniu/http/httputil.v1"
)

var (
	// server error
	ErrFeatureSetExist    = httputil.NewError(http.StatusBadRequest, "feature set is already exist")
	ErrInvalidDeviceID    = errors.New("invalid device id")
	ErrInvalidFeautres    = httputil.NewError(http.StatusBadRequest, "invalid features in request")
	ErrFeatureSetNotFound = httputil.NewError(http.StatusNotFound, "feature set not found")
	ErrInvalidSetState    = httputil.NewError(http.StatusBadRequest, "invalid set state")
)
