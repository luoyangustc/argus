package uri

import "strings"

type Uri string

func NewURI(str string) Uri { return Uri(str) }

func (u Uri) ToString() string { return string(u) }

type UriType int

const (
	NONE  UriType = 0
	FILE  UriType = 1 // file://文件系统绝对路径
	HTTP  UriType = 2 // http://host:port/xx or https://host:port/xx
	STS   UriType = 3 // sts://ip:port/xx or sts://ip:port/xx
	QINIU UriType = 4 // qiniu://uid@zone/bucket/key
	DATA  UriType = 5 // data:application/octet-stream;base64,xxx
)

func TypeOf(uri Uri) UriType {
	str := uri.ToString()
	if strings.HasPrefix(str, "file://") {
		return FILE
	} else if strings.HasPrefix(str, "http://") || strings.HasPrefix(str, "https://") {
		return HTTP
	} else if strings.HasPrefix(str, "sts://") {
		return STS
	} else if strings.HasPrefix(str, "qiniu://") {
		return QINIU
	} else if strings.HasPrefix(str, "data:application/octet-stream;base64,") {
		return DATA
	}
	return NONE
}

func SchemeOf(uri Uri) string {
	str := uri.ToString()
	if strings.HasPrefix(str, "file://") {
		return "file"
	} else if strings.HasPrefix(str, "http://") || strings.HasPrefix(str, "https://") {
		return "http"
	} else if strings.HasPrefix(str, "sts://") {
		return "sts"
	} else if strings.HasPrefix(str, "qiniu://") {
		return "qiniu"
	} else if strings.HasPrefix(str, "data:application/octet-stream;base64,") {
		return "data"
	}
	return ""
}
