package facec

import (
	"context"
	"net/http"

	"github.com/pkg/errors"
	"github.com/qiniu/http/httputil.v1"
	"github.com/qiniu/log.v1"
	"github.com/qiniu/xlog.v1"
)

var ErrImageNumExceeded = httputil.NewError(http.StatusBadRequest, "number of images exceeded 10")
var ErrEuidEmpty = httputil.NewError(http.StatusBadRequest, "euid should not empty")
var ErrGroupNotExists = httputil.NewError(http.StatusBadRequest, "group not exists")
var ErrInvalidArgs = httputil.NewError(http.StatusBadRequest, "Invalid arguments")

// 图片预处理失败的时候返回的错误码
var StatusCodeImageProcessError int64 = 415

// 最大图片数目
const MAXImageNum = 10

var Wrap = errors.Wrap

func WrapWithLog(ctx context.Context, err error, message string) error {
	err2 := Wrap(err, message)
	log.Std.Output(xlog.FromContextSafe(ctx).ReqId(), log.Lerror, 2, err2.Error())
	return err2
}

func ErrBadRequest(msg string) error {
	return httputil.NewError(http.StatusBadRequest, "group not exists")
}

const MODEL_VERSION = "1"
