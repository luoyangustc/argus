package httputil

import (
	"encoding/json"
	"io"
	"net/http"
	"reflect"
	"strconv"
	"strings"
	"syscall"
	"context"

	"github.com/qiniu/errors"
	"github.com/qiniu/log.v1"
	"github.com/qiniu/rpc.v1"
)

// ---------------------------------------------------------------------------
// func Reply

func Reply(w http.ResponseWriter, code int, data interface{}) {

	msg, err := json.Marshal(data)
	if err != nil {
		Error(w, err)
		return
	}

	h := w.Header()
	h.Set("Content-Length", strconv.Itoa(len(msg)))
	h.Set("Content-Type", "application/json")
	w.WriteHeader(code)
	w.Write(msg)
}

func ReplyWith(w http.ResponseWriter, code int, bodyType string, msg []byte) {

	h := w.Header()
	h.Set("Content-Length", strconv.Itoa(len(msg)))
	h.Set("Content-Type", bodyType)
	w.WriteHeader(code)
	w.Write(msg)
}

func ReplyWithStream(w http.ResponseWriter, code int, bodyType string, body io.Reader, bytes int64) {

	h := w.Header()
	h.Set("Content-Length", strconv.FormatInt(bytes, 10))
	h.Set("Content-Type", bodyType)
	w.WriteHeader(code)
	io.Copy(w, body) // don't use io.CopyN: if you need, call io.LimitReader(body, bytes) by yourself
}

func ReplyWithCode(w http.ResponseWriter, code int) {

	if code < 400 {
		h := w.Header()
		h.Set("Content-Length", "2")
		h.Set("Content-Type", "application/json")
		w.WriteHeader(code)
		w.Write(emptyObj)
	} else {
		err := http.StatusText(code)
		if err == "" {
			err = "E" + strconv.Itoa(code)
		}
		ReplyErr(w, code, err)
	}
}

var emptyObj = []byte{'{', '}'}

// ---------------------------------------------------------------------------
// func Error

var HttpCodeOf = func(err error) int {
	return 599
}

func Error(w http.ResponseWriter, err error) {

	detail := errors.Detail(err)
	code, errStr := DetectError(err)
	replyErr(2, w, code, errStr, detail)
}

func ReplyErr(w http.ResponseWriter, code int, err string) {

	replyErr(2, w, code, err, err)
}

type httpCoder interface {
	HttpCode() int
}

func DetectCode(err error) int {

	err = errors.Err(err)
	if e, ok := err.(*ErrorInfo); ok {
		return e.Code
	} else if e, ok := err.(rpc.RespError); ok {
		return e.HttpCode()
	} else if e, ok := err.(httpCoder); ok {
		return e.HttpCode()
	}
	switch err {
	case syscall.EINVAL:
		return 400
	case syscall.ENOENT:
		return 612 // no such entry
	case syscall.EEXIST:
		return 614 // entry exists
	}
	return HttpCodeOf(err)
}

type rpcError interface {
	Error() string
	HttpCode() int
}

func DetectError(err error) (code int, desc string) {

	err = errors.Err(err)
	if e, ok := err.(*ErrorInfo); ok {
		return e.Code, e.Err
	} else if e, ok := err.(*rpc.ErrorInfo); ok {
		return e.Code, e.Err
	} else if e, ok := err.(rpcError); ok {
		return e.HttpCode(), e.Error()
	}
	switch err {
	case syscall.EINVAL:
		return 400, "invalid arguments"
	case syscall.ENOENT:
		return 612, "no such entry"
	case syscall.EEXIST:
		return 614, "entry exists"
	case context.Canceled:
		return 499, context.Canceled.Error()
	}
	return HttpCodeOf(err), err.Error()
}

type errorRet struct {
	Error string `json:"error"`
}

func replyErr(lvl int, w http.ResponseWriter, code int, err, detail string) {

	logWithReqid(lvl+1, w.Header().Get("X-Reqid"), detail)

	msg, _ := json.Marshal(errorRet{err})

	h := w.Header()
	h.Set("Content-Length", strconv.Itoa(len(msg)))
	h.Set("Content-Type", "application/json")
	w.WriteHeader(code)
	w.Write(msg)
}

func logWithReqid(lvl int, reqid string, str string) {
	str = strings.Replace(str, "\n", "\n["+reqid+"]", -1)
	log.Std.Output(reqid, log.Lwarn, lvl+1, str)
}

// ---------------------------------------------------------------------------
// type ErrorInfo

type ErrorInfo struct {
	Err  string
	Code int
}

func NewError(code int, err string) *ErrorInfo {
	return &ErrorInfo{err, code}
}

func (e *ErrorInfo) Error() string {
	return e.Err
}

// ---------------------------------------------------------------------------

const (
	StatusGracefulQuit = 570 // 停止服务中（重启中）
	StatusOverload     = 571 // 过载保护阶段（处理部分请求，其余返回 571 重试码)
	StatusAbnormal     = 572 // 认为自身工作不正常时，比如长时间没法拿到数据
)

var (
	ErrGracefulQuit = NewError(StatusGracefulQuit, "graceful quit")
	ErrOverload     = NewError(StatusOverload, "overload")
)

// ---------------------------------------------------------------------------

func GetCloseNotifier(w http.ResponseWriter) (cn http.CloseNotifier, ok bool) {

	if cn, ok = w.(http.CloseNotifier); ok {
		return
	}

	v := reflect.ValueOf(w)
	v = reflect.Indirect(v)
	for v.Kind() == reflect.Struct {
		if fv := v.FieldByName("ResponseWriter"); fv.IsValid() {
			if cn, ok = fv.Interface().(http.CloseNotifier); ok {
				return
			}
			v = reflect.Indirect(fv)
		} else {
			break
		}
	}
	return
}

type fakeCloseNotifier struct{}

func (fcn fakeCloseNotifier) CloseNotify() <-chan bool {

	c := make(chan bool, 1)
	return c
}

func GetCloseNotifierSafe(w http.ResponseWriter) http.CloseNotifier {

	if cn, ok := GetCloseNotifier(w); ok {
		return cn
	}
	return fakeCloseNotifier{}
}

// ---------------------------------------------------------------------------

type RequestCanceler interface {
	CancelRequest(req *http.Request)
}

func GetRequestCanceler(tp http.RoundTripper) (rc RequestCanceler, ok bool) {

	v := reflect.ValueOf(tp)

subfield:
	// panic if the Field is unexported (but this can be detected in developing)
	if rc, ok = v.Interface().(RequestCanceler); ok {
		return
	}
	v = reflect.Indirect(v)
	if v.Kind() == reflect.Struct {
		for i := v.NumField() - 1; i >= 0; i-- {
			sv := v.Field(i)
			if sv.Kind() == reflect.Interface {
				sv = sv.Elem()
			}
			if sv.MethodByName("RoundTrip").IsValid() {
				v = sv
				goto subfield
			}
		}
	}
	return
}

// ---------------------------------------------------------------------------

func GetHijacker(w http.ResponseWriter) (hj http.Hijacker, ok bool) {

	if hj, ok = w.(http.Hijacker); ok {
		return
	}

	v := reflect.ValueOf(w)
	v = reflect.Indirect(v)
	for v.Kind() == reflect.Struct {
		if fv := v.FieldByName("ResponseWriter"); fv.IsValid() {
			if hj, ok = fv.Interface().(http.Hijacker); ok {
				return
			}
			if fv.Kind() == reflect.Interface {
				fv = fv.Elem()
			}
			v = reflect.Indirect(fv)
		} else {
			break
		}
	}
	return
}

// ---------------------------------------------------------------------------

func Flusher(w http.ResponseWriter) (f http.Flusher, ok bool) {

	if f, ok = w.(http.Flusher); ok {
		return
	}

	v := reflect.ValueOf(w)
	v = reflect.Indirect(v)
	for v.Kind() == reflect.Struct {
		if fv := v.FieldByName("ResponseWriter"); fv.IsValid() {
			if f, ok = fv.Interface().(http.Flusher); ok {
				return
			}
			if fv.Kind() == reflect.Interface {
				fv = fv.Elem()
			}
			v = reflect.Indirect(fv)
		} else {
			break
		}
	}
	return
}

// ---------------------------------------------------------------------------
