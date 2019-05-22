package manager

const (
	CONTENT_TYPE   = "Content-Type"
	CONTENT_LENGTH = "Content-Length"
	CT_JSON        = "application/json"
	CT_STREAM      = "application/octet-stream"
)

// IsJsonContent ...
func IsJsonContent(contentType string) bool {
	switch contentType {
	case CT_JSON, "application/json; charset=UTF-8", "application/json; charset=utf-8":
		return true
	default:
		return false
	}
}

type M map[string]interface{}
