package proto

import (
	"net/http"

	httputil "github.com/qiniu/http/httputil.v1"
)

var (
	ErrUnauthorized = httputil.NewError(http.StatusUnauthorized, "unauthorized")
	ErrForbidden    = httputil.NewError(http.StatusForbidden, "forbidden")
	ErrGenId        = httputil.NewError(http.StatusBadRequest, "cannot generate id")

	// entry error
	ErrEmptyId           = httputil.NewError(http.StatusBadRequest, "empty id")
	ErrEmptyIds          = httputil.NewError(http.StatusBadRequest, "empty ids")
	ErrInvalidSuggestion = httputil.NewError(http.StatusBadRequest, "invalid suggestion")
	ErrInvalidMarker     = httputil.NewError(http.StatusBadRequest, "invalid marker")

	// user error
	ErrUserNotExist      = httputil.NewError(http.StatusBadRequest, "user does not exist")
	ErrEmptyUsername     = httputil.NewError(http.StatusBadRequest, "empty username")
	ErrEmptyPassword     = httputil.NewError(http.StatusBadRequest, "empty password")
	ErrEmptyRoles        = httputil.NewError(http.StatusBadRequest, "empty roles")
	ErrCannotCreateAdmin = httputil.NewError(http.StatusBadRequest, "cannot create admin")
	ErrCannotDeleteAdmin = httputil.NewError(http.StatusBadRequest, "cannot delete admin")
	ErrCannotUpdateAdmin = httputil.NewError(http.StatusBadRequest, "cannot update admin")
	ErrInvalidRole       = httputil.NewError(http.StatusBadRequest, "invalid role")
	ErrUserExist         = httputil.NewError(http.StatusBadRequest, "user already exists")
	ErrIncorrectPwd      = httputil.NewError(http.StatusBadRequest, "incorrect password")

	// set error
	ErrSetNotExist          = httputil.NewError(http.StatusBadRequest, "set does not exist")
	ErrEntryNotExist        = httputil.NewError(http.StatusBadRequest, "entry does not exist")
	ErrSetHistoryNotExist   = httputil.NewError(http.StatusBadRequest, "set history does not exist")
	ErrInvalidScene         = httputil.NewError(http.StatusBadRequest, "invalid scene")
	ErrInvalidType          = httputil.NewError(http.StatusBadRequest, "invalid type")
	ErrEmptyName            = httputil.NewError(http.StatusBadRequest, "empty name")
	ErrEmptyType            = httputil.NewError(http.StatusBadRequest, "empty type")
	ErrEmptyScenes          = httputil.NewError(http.StatusBadRequest, "empty scenes")
	ErrEmptyUri             = httputil.NewError(http.StatusBadRequest, "empty uri")
	ErrMonitorIntervalSmall = httputil.NewError(http.StatusBadRequest, "monitor interval too small")
	ErrExistUri             = httputil.NewError(http.StatusBadRequest, "uri exists in other set")
	ErrExistName            = httputil.NewError(http.StatusBadRequest, "name exists in other set")
	ErrInvalidMimeType      = httputil.NewError(http.StatusBadRequest, "invalid mime_type")
	ErrEmptyMimeTypes       = httputil.NewError(http.StatusBadRequest, "empty mime_types")
	ErrSetNotRunning        = httputil.NewError(http.StatusBadRequest, "set is not running")
	ErrSetNotStopped        = httputil.NewError(http.StatusBadRequest, "set is already running, completed or deleted")
	ErrInvalidFile          = httputil.NewError(http.StatusBadRequest, "file is not uploaded, invalid or does not have valid content")
	ErrInvalidCutInterval   = httputil.NewError(http.StatusBadRequest, "invalid cut interval")

	// resources error
	ErrEmptyResource   = httputil.NewError(http.StatusBadRequest, "empty resource")
	ErrInvalidResource = httputil.NewError(http.StatusBadRequest, "invalid resource")
	ErrResourceExist   = httputil.NewError(http.StatusBadRequest, "resource already exists")
	ErrEmptyUrls       = httputil.NewError(http.StatusBadRequest, "empty urls")

	// video cut error
	ErrVideoCutNotExist = httputil.NewError(http.StatusBadRequest, "video cut does not exist")
)
