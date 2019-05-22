package gate

import (
	"encoding/base64"
	"net/url"
	"strconv"
	"strings"
)

func improveURI(uri string, uid uint32) (string, error) {
	_url, err := url.Parse(uri)
	if err != nil {
		return uri, err
	}
	if _url.Scheme != "qiniu" {
		return uri, nil
	}
	_url.User = url.User(strconv.Itoa(int(uid)))
	return _url.String(), nil
}

////////////////////////////////////////////////////////////////////////////////

const (
	_DataURIPrefix = "data:application/octet-stream;base64,"
)

func isBadURI(uri string) bool {
	switch {
	case strings.HasPrefix(uri, "file://"):
		return false
	case strings.HasPrefix(uri, "http://") || strings.HasPrefix(uri, "https://"):
		return false
	case strings.HasPrefix(uri, "sts://"):
		return false
	case strings.HasPrefix(uri, "qiniu://"):
		return false
	case strings.HasPrefix(uri, "data:application/octet-stream;base64,"):
		return false
	default:
		return true
	}
}

func isDataURI(uri string) bool {
	return strings.HasPrefix(uri, _DataURIPrefix)
}

func isStsURI(uri string) bool {
	return strings.HasPrefix(uri, "sts://")
}

func encodeDataURI(src []byte) string {
	return _DataURIPrefix + base64.StdEncoding.EncodeToString(src)
}

func decodeDataURI(src string) ([]byte, error) {
	return base64.StdEncoding.DecodeString(strings.TrimPrefix(src, _DataURIPrefix))
}
