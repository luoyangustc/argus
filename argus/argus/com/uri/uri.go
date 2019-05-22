package uri

const (
	DataURIPrefix = "data:application/octet-stream;base64,"
)

// STRING string
type STRING string

func (s STRING) String() string { return string(s) }

// GoString ...
func (s STRING) GoString() string {
	if len(s) > 256 {
		return string(s)[:253] + "..."
	}
	return string(s)
}
