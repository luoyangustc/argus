package segment

import (
	"errors"
	"net/http"
	"strings"

	"github.com/qiniu/http/httputil.v1"
)

//
var (
	ErrRetry      = errors.New("need retry")
	ErrBadRequest = httputil.NewError(http.StatusBadRequest, "bad request")

	ErrConflictToken = httputil.NewError(http.StatusConflict, "conflict token")

	ErrInvalidMode      = httputil.NewError(http.StatusBadRequest, "invalid mode, allow mode is [0, 1]")
	ErrInvalidInterval  = httputil.NewError(http.StatusBadRequest, "invalid interval, allow mode is [0, 10]")
	ErrInvalidStartTime = httputil.NewError(http.StatusBadRequest, "invalid starttime")
	ErrInvalidDuration  = httputil.NewError(http.StatusBadRequest, "invalid duration")

	// Error returned from gen_pic cmd
	ErrInvalidParameters = httputil.NewError(http.StatusBadRequest, "invalid parameters")
	ErrCannotFindVideo   = httputil.NewError(http.StatusFailedDependency, "cannot find the video")
	ErrCannotOpenFile    = httputil.NewError(http.StatusBadRequest, "cannot open the file")
	ErrCannotAllowMemory = httputil.NewError(http.StatusInternalServerError, "cannot allow memory")

	ErrCanceled = httputil.NewError(499, "context canceled")
)

func GenSegmentCmdError(err error) error {
	switch {
	case strings.Compare(err.Error(), "exit status 1") == 0:
		return ErrInvalidParameters
	case strings.Compare(err.Error(), "exit status 2") == 0,
		strings.Compare(err.Error(), "exit status 3") == 0:
		return ErrCannotFindVideo
	case strings.Compare(err.Error(), "exit status 4") == 0:
	case strings.Compare(err.Error(), "exit status 5") == 0,
		strings.Compare(err.Error(), "exit status 6") == 0,
		strings.Compare(err.Error(), "exit status 7") == 0,
		strings.Compare(err.Error(), "exit status 9") == 0,
		strings.Compare(err.Error(), "exit status 10") == 0,
		strings.Compare(err.Error(), "exit status 11") == 0:
		return ErrCannotOpenFile
	case strings.Compare(err.Error(), "exit status 8") == 0:
		return ErrCannotAllowMemory
	}

	return nil
}
