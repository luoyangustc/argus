package util

import (
	"fmt"
	"net/url"
	"regexp"
)

var pathRegex = regexp.MustCompile(`/(.*?)/(.*)`)

func ParseQiniuURI(uri string) (bucket string, key string, err error) {
	// u: Scheme: "qiniu", Host: "z0", Path: "/test/1.png",
	u, err := url.Parse(uri)
	if err != nil {
		return
	}
	subStr := pathRegex.FindStringSubmatch(u.Path)
	if len(subStr) != 3 {
		err = fmt.Errorf("invalid qiniu uri")
		return
	}
	bucket = subStr[1]
	key = subStr[2]
	return
}
