package qnhttp

import "math/rand"

type httpError struct {
	status int
	errMsg string
}

var errorRepo = []httpError{
	httpError{608, "FileModified   = 608 // RS: 文件被修改（see fs.GetIfNotModified）"},
	httpError{612, "NoSuchEntry    = 612 // RS: 指定的 Entry 不存在或已经 Deleted"},
	httpError{614, "EntryExists    = 614 // RS: 要创建的 Entry 已经存在"},
	httpError{630, "TooManyBuckets = 630 // RS: 创建的 Bucket 个数过多"},
	httpError{631, "NoSuchBucket   = 631 // RS: 指定的 Bucket 不存在"},
	httpError{400, "Bad Request"},
	httpError{500, "Interal Server Error"},
	httpError{502, "Bad Gateway"},
	httpError{504, "Gateway Timeout"},
}

// Select error code randomly
func Pick() int {
	return errorRepo[rand.Intn(len(errorRepo))].status
}
