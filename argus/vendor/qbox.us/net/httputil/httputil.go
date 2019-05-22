// [DEPRECATED] 本package已经迁移到 "github.com/qiniu/http/httputil.v1"
package httputil

import (
	"encoding/json"
	"io"
	"net/http"
	"os"
	"strconv"

	"github.com/qiniu/log.v1"
	"qbox.us/api"
	"qbox.us/errors"

	qhttputil "github.com/qiniu/http/httputil.v1"
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

func ReplyWithCode(w http.ResponseWriter, code int) {

	if code < 400 {
		h := w.Header()
		h.Set("Content-Length", "2")
		h.Set("Content-Type", "application/json")
		w.WriteHeader(code)
		w.Write([]byte(`{}`))
	} else {
		err := api.NewError(code)
		Error(w, err)
	}
}

// ---------------------------------------------------------------------------

//
// TODO: 此函数可能被废除，请调用 httputil.ReplyBinary
//
func ReplyWithBinary(w http.ResponseWriter, body io.Reader, bytes int64) {

	h := w.Header()
	h.Set("Content-Length", strconv.FormatInt(bytes, 10))
	h.Set("Content-Type", "application/octet-stream")
	w.WriteHeader(http.StatusOK)
	io.CopyN(w, body, bytes)
}

//
// TODO: 此函数可能被废除，请调用 httputil.ReplyChunked
//
func ReplyWithBinaryChunked(w http.ResponseWriter, body io.Reader) {

	h := w.Header()
	h.Set("Content-Type", "application/octet-stream")
	w.WriteHeader(http.StatusOK)
	io.Copy(w, body)
}

//
// TODO: 此函数可能被废除，请调用 httputil.ReplyErr，主要是交换了下参数次序
//
func ReplyError(w http.ResponseWriter, err string, code int) {

	ReplyErr(w, code, err)
}

// ---------------------------------------------------------------------------
// func ReplyFile

func ReplyFile(w http.ResponseWriter, file string) {

	f, err := os.Open(file)
	if err != nil {
		err = errors.Info(err, "httputil.ReplyFile", "open failed", file).Detail(err)
		Error(w, err)
		return
	}
	defer f.Close()

	fi, err := f.Stat()
	if err != nil {
		err = errors.Info(err, "httputil.ReplyFile", "stat failed", file).Detail(err)
		Error(w, err)
		return
	}

	// detect content type
	bs := make([]byte, 1024)
	n, err := f.Read(bs)
	if err != nil {
		err = errors.Info(err, "httputil.ReplyFile", "read file for detect mimeType failed", file).Detail(err)
		return
	}
	mimeType := http.DetectContentType(bs[:n])

	// rewind to file's begin
	if _, err = f.Seek(0, os.SEEK_SET); err != nil {
		err = errors.Info(err, "httputil.ReplyFile", "seek file failed", file).Detail(err)
		return
	}

	ReplyBinaryEx(w, f, fi.Size(), mimeType)
}

func ReplyBinary(w http.ResponseWriter, body io.Reader, bytes int64) {

	h := w.Header()
	h.Set("Content-Length", strconv.FormatInt(bytes, 10))
	h.Set("Content-Type", "application/octet-stream")
	w.WriteHeader(http.StatusOK)
	io.Copy(w, body) // don't use io.CopyN: if you need, call io.LimitReader(body, bytes) by yourself
}

func ReplyBinaryEx(w http.ResponseWriter, body io.Reader, bytes int64, mimeType string) {

	h := w.Header()
	h.Set("Content-Length", strconv.FormatInt(bytes, 10))
	h.Set("Content-Type", mimeType)
	w.WriteHeader(http.StatusOK)
	io.Copy(w, body) // don't use io.CopyN: if you need, call io.LimitReader(body, bytes) by yourself
}

func ReplyChunked(w http.ResponseWriter, body io.Reader) {

	h := w.Header()
	h.Set("Content-Type", "application/octet-stream")
	w.WriteHeader(http.StatusOK)
	io.Copy(w, body)
}

// ---------------------------------------------------------------------------
// func Error

type errorRet struct {
	Error string `json:"error"`
}

func ReplyErr(w http.ResponseWriter, code int, err string) {

	replyErr(w, code, err, err)
}

func replyErr(w http.ResponseWriter, code int, err, detail string) {

	log.Std.Output(w.Header().Get("X-Reqid"), log.Lwarn, 2, detail)

	msg, _ := json.Marshal(errorRet{err})

	h := w.Header()
	h.Set("Content-Length", strconv.Itoa(len(msg)))
	h.Set("Content-Type", "application/json")
	w.WriteHeader(code)
	w.Write(msg)
}

func Error(w http.ResponseWriter, err error) {

	detail := errors.Detail(err)
	code, errStr := DetectError(err)
	replyErr(w, code, errStr, detail)
}

func DetectError(err error) (code int, desc string) {

	err = errors.Err(err)
	if e, ok := err.(*ErrorInfo); ok {
		return e.Code, e.Err
	} else if e, ok := err.(*rpc.ErrorInfo); ok {
		return e.Code, e.Err
	} else if e, ok := err.(*qhttputil.ErrorInfo); ok {
		return e.Code, e.Err
	}

	// old style
	return api.HttpCode(err), err.Error()
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
