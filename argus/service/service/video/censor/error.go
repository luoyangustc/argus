package censor

import (
	"net/http"

	httputil "github.com/qiniu/http/httputil.v1"

	"qiniu.com/argus/sdk/video"
	. "qiniu.com/argus/service/service"
)

func formatError(err error) error {
	if _, ok := err.(DetectErrorer); ok {
		return err
	}
	switch err {
	case video.GenpicErrCannotFindVideo:
		return ErrVideoType("cannot find video") //文件可打开但无法解析,如音频文件
	case video.GenpicErrCannotOpenFile:
		return ErrUriFetchFailed("fetch uri failed") //文件无法打开或无效uri
	}
	switch err2 := err.(type) {
	case *httputil.ErrorInfo:
		switch err2.Code {
		case http.StatusBadRequest:
			return ErrArgs(err.Error())
		default:
			return ErrInternal(err.Error())
		}
	default:
		return ErrInternal(err.Error())
	}
}
