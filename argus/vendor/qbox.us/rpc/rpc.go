package rpc

import (
	"encoding/json"
	"io"
	"net/http"
	"os"
	"strconv"
	"strings"
	"syscall"
	"time"

	"github.com/qiniu/http/httputil.v1"
	"github.com/qiniu/log.v1"
	"github.com/qiniu/xlog.v1"
	"qbox.us/errors"
	"qiniupkg.com/x/url.v7"
)

// -------------------------------------------------------

type RangeReader interface {
	RangeRead(w io.Writer, from, to int64) (err error)
}

type ReadSeeker2RangeReader struct {
	io.ReadSeeker
}

func (r ReadSeeker2RangeReader) RangeRead(w io.Writer, from, to int64) (err error) {
	if from > to {
		return syscall.EINVAL
	}
	_, err = r.Seek(from, os.SEEK_SET)
	if err != nil {
		return
	}
	_, err = io.CopyN(w, r, to-from)
	return
}

// -------------------------------------------------------
// ResponseWriter

//
// TODO: 此类可能被废除，建议使用 qbox.us/net/httputil 中相对应的函数。
//
type ResponseWriter struct {
	http.ResponseWriter
}

func (w ResponseWriter) ReplyWith(code int, data interface{}) {
	var msg []byte
	if data != nil {
		msg, _ = json.Marshal(data)
	}
	h := w.Header()
	h.Set("Content-Length", strconv.Itoa(len(msg)))
	h.Set("Content-Type", "application/json")
	h.Set("Cache-Control", "no-store")
	w.WriteHeader(code)
	w.Write(msg)
}

func (w ResponseWriter) ReplyWithCode(code int) {
	h := w.Header()
	h.Set("Content-Length", "0")
	h.Set("Content-Type", "application/json")
	w.WriteHeader(code)
}

func (w ResponseWriter) ReplyWithParam(code int, bodyType string, msg []byte) {
	h := w.Header()
	h.Set("Content-Length", strconv.Itoa(len(msg)))
	h.Set("Content-Type", bodyType)
	w.WriteHeader(code)
	w.Write(msg)
}

func (w ResponseWriter) ReplyWithBinary(body io.Reader, bytes int64) {
	h := w.Header()
	h.Set("Content-Length", strconv.FormatInt(bytes, 10))
	h.Set("Content-Type", "application/octet-stream")
	w.WriteHeader(200)
	io.CopyN(w, body, bytes)
}

func (w ResponseWriter) replyErr(code int, err, detail string) {

	log.Std.Output(w.Header().Get("X-Reqid"), log.Lwarn, 2, detail)
	w.ReplyWith(code, &ErrorRet{err})
}

func (w ResponseWriter) ReplyWithError(code int, err error) {

	detail := errors.Detail(err)
	w.replyErr(code, err.Error(), detail)
}

func (w ResponseWriter) ReplyJSONP(callback string, data interface{}) {
	var msg1 []byte
	if data != nil {
		msg1, _ = json.Marshal(data)
	}
	msg := append([]byte(callback), '(')
	msg = append(msg, msg1...)
	msg = append(msg, ')')

	h := w.Header()
	h.Set("Content-Length", strconv.Itoa(len(msg)))
	h.Set("Content-Type", "text/javascript")
	h.Set("Cache-Control", "no-store")
	w.WriteHeader(200)
	w.Write(msg)
}

func (w ResponseWriter) Reply(data interface{}) {
	w.ReplyWith(200, data)
}

func (w ResponseWriter) ReplyError(code int, err string) {

	w.replyErr(code, err, err)
}

type Metas struct {
	ETag            string
	XETag           string
	MimeType        string // "application/octet-stream"
	DispositionType string // "attachment", "inline"
	FileName        string
	CacheControl    string
	Expires         string
	Is404           bool
	LastModified    time.Time
}

func getETag(etag string) string {
	last := len(etag)
	if last > 1 && etag[0] == '"' && etag[last-1] == '"' {
		return etag[1 : last-1]
	}
	return etag
}

func (w ResponseWriter) ReplyFile(f io.Reader, fsize int64, meta *Metas, req *http.Request) (err error) {

	if Handle304(w, meta, req) {
		return
	}

	return w.ReplyFileNo304(f, fsize, meta, req)
}

func (w ResponseWriter) ReplyFileNo304(f io.Reader, fsize int64, meta *Metas, req *http.Request) (err error) {

	xl := xlog.New(w.ResponseWriter, req)

	h := w.Header()
	SetHeaders(h, meta, req)

	if fsize < 0 {
		w.WriteHeader(200)
		if req.Method != "HEAD" {
			_, err = io.Copy(w, f)
		}
	} else {
		h.Set("Content-Length", strconv.FormatInt(fsize, 10))
		w.WriteHeader(200)
		if req.Method != "HEAD" {
			_, err = io.CopyN(w, f, fsize)
		}
	}
	if err != nil {
		xl.Info("ReplyFile: io.Copy failed =>", errors.Detail(err))
	}
	return
}

func (w ResponseWriter) ReplyRange(f RangeReader, fsize int64, meta *Metas, req *http.Request) (err error) {

	if Handle304(w, meta, req) {
		return
	}

	return w.ReplyRangeNo304(f, fsize, meta, req)
}

func (w ResponseWriter) ReplyRangeNo304(f RangeReader, fsize int64, meta *Metas, req *http.Request) (err error) {

	xl := xlog.New(w.ResponseWriter, req)

	h := w.Header()
	SetHeaders(h, meta, req)

	if rg1, ok := req.Header["Range"]; ok && !strings.HasSuffix(meta.ETag, ".gz") && !meta.Is404 {
		from, to, ok := ParseOneRange(rg1[0], fsize)
		if !ok {
			w.ReplyWithCode(http.StatusRequestedRangeNotSatisfiable)
			return
		}
		rg := "bytes " + strconv.FormatInt(from, 10) + "-" + strconv.FormatInt(to-1, 10) + "/" + strconv.FormatInt(fsize, 10)
		h.Set("Content-Length", strconv.FormatInt(to-from, 10))
		h.Set("Content-Range", rg)
		w := newHeaderResponseWrite(w)
		w.WriteHeader(206)
		if req.Method != "HEAD" {
			err = f.RangeRead(w, from, to)
		}
		w.Done(err)
	} else {
		if !strings.HasSuffix(meta.ETag, ".gz") {
			h.Set("Content-Length", strconv.FormatInt(fsize, 10))
		}
		w := newHeaderResponseWrite(w)
		if meta.Is404 {
			w.WriteHeader(404)
		} else {
			w.WriteHeader(200)
		}
		if req.Method != "HEAD" {
			err = f.RangeRead(w, 0, fsize)
		}
		w.Done(err)
	}
	if err != nil {
		xl.Info("ReplyRange: RangeRead failed =>", errors.Detail(err))
	}
	return
}

func (w ResponseWriter) ReplyRange2(f io.ReadSeeker, fsize int64, meta *Metas, req *http.Request) (err error) {
	return w.ReplyRange(ReadSeeker2RangeReader{f}, fsize, meta, req)
}

func SetHeaders(h http.Header, meta *Metas, req *http.Request) {
	if meta.MimeType == "" {
		meta.MimeType = "application/octet-stream"
	}

	h.Set("Accept-Ranges", "bytes")
	h.Set("Content-Transfer-Encoding", "binary")
	h.Set("Content-Type", meta.MimeType)
	if meta.ETag != "" {
		h.Set("ETag", "\""+meta.ETag+"\"")
	}
	if meta.XETag != "" {
		h.Set("X-ETag", "\""+meta.XETag+"\"")
	}
	h.Set("Last-Modified", meta.LastModified.UTC().Format(http.TimeFormat))
	if meta.DispositionType != "" {
		// https: //jira.qiniu.io/browse/KODO-1202
		h.Set("Content-Disposition", meta.DispositionType+"; filename*= UTF-8' '"+url.PathEscape(meta.FileName))
	}
	if meta.CacheControl != "" {
		h.Set("Cache-Control", meta.CacheControl)
	}
	if meta.Expires != "" {
		h.Set("Expires", meta.Expires)
	}
}

func Handle304(w http.ResponseWriter, meta *Metas, req *http.Request) (if304 bool) {
	xl := xlog.New(w, req)
	h := w.Header()

	if304 = false
	// http://www.w3.org/Protocols/rfc2616/rfc2616-sec10.html#sec10.3.5
	if ims := req.Header.Get("If-Modified-Since"); ims != "" {
		if t, err := http.ParseTime(ims); err == nil {
			if meta.LastModified.Truncate(time.Second).After(t) {
				// IMS is in second, so meta.LastModified must be trucated before comparing.
				return false
			} else {
				if304 = true
			}
		} else {
			xl.Info("ReplyRange: Parsing If-Modified-Since failed:", err)
		}
	}
	if meta.ETag != "" {
		if etag := req.Header.Get("If-None-Match"); etag != "" {
			if getETag(etag) != meta.ETag {
				return false
			} else {
				if304 = true
				h.Set("ETag", "\""+meta.ETag+"\"")
			}
		}
	}
	if !if304 {
		return false
	}
	// 如果If-Modified-Since和If-None-Match都有，则都满足时才能返回304
	xl.Println("ReplyRange: 304 Not Modified")
	if meta.CacheControl != "" {
		h.Set("Cache-Control", meta.CacheControl)
	}
	if meta.Expires != "" {
		h.Set("Expires", meta.Expires)
	}
	w.WriteHeader(304)
	return true
}

// -------------------------------------------------------
// Request

type Request struct {
	*http.Request
	Query []string
}

// -------------------------------------------------------
// Handler

type Handler interface {
	ServeHTTP(w ResponseWriter, req Request)
}

// -------------------------------------------------------
// ServeMux

type ServeMux struct {
	handlers map[string]func(w ResponseWriter, req Request)
}

func NewServeMux() *ServeMux {
	return &ServeMux{make(map[string]func(w ResponseWriter, req Request))}
}

func (p *ServeMux) Handle(method string, handler Handler) {
	p.handlers[method] = func(w ResponseWriter, req Request) { handler.ServeHTTP(w, req) }
}

func (p *ServeMux) HandleFunc(method string, handler func(w ResponseWriter, req Request)) {
	p.handlers[method] = handler
}

func (p *ServeMux) RegisterHandlers(mux *http.ServeMux) {
	mux.HandleFunc("/", func(w http.ResponseWriter, req *http.Request) {
		p.process(ResponseWriter{w}, req)
	})
}

func (p *ServeMux) process(w ResponseWriter, req *http.Request) {
	query := strings.Split(req.URL.Path[1:], "/")
	if handler, ok := p.handlers[query[0]]; ok {
		handler(w, Request{req, query})
	} else {
		w.ReplyError(400, "Bad method")
	}
}

// -------------------------------------------------------
// Helper

var DefaultServeMux = NewServeMux()

func Handle(method string, handler Handler) {
	DefaultServeMux.Handle(method, handler)
}

func HandleFunc(method string, handler func(w ResponseWriter, req Request)) {
	DefaultServeMux.HandleFunc(method, handler)
}

func Run(host string, mux *http.ServeMux) error {
	if mux == nil {
		mux = http.DefaultServeMux
	}
	DefaultServeMux.RegisterHandlers(mux)
	return http.ListenAndServe(host, mux)
}

//-------------------------------

type headerResponseWriter struct {
	ResponseWriter
	headerCode int
	firstWrite bool
}

func newHeaderResponseWrite(w ResponseWriter) *headerResponseWriter {
	return &headerResponseWriter{
		ResponseWriter: w,
		headerCode:     http.StatusOK,
		firstWrite:     true,
	}
}

func (w *headerResponseWriter) Write(p []byte) (n int, err error) {
	if w.firstWrite && len(p) > 0 {
		w.firstWrite = false
		w.ResponseWriter.WriteHeader(w.headerCode)
	}
	return w.ResponseWriter.Write(p)
}

func (w *headerResponseWriter) WriteHeader(code int) {
	w.headerCode = code
	if w.headerCode/100 != 2 {
		w.ResponseWriter.WriteHeader(code)
		w.firstWrite = false
	}
	return
}
func (w *headerResponseWriter) replyErr(err error) {
	code := httputil.DetectCode(err)
	w.ResponseWriter.ReplyWithError(code, err)
	return
}

func (w *headerResponseWriter) Done(err error) {
	if w.firstWrite {
		if err == nil {
			w.ResponseWriter.WriteHeader(w.headerCode)
		} else {
			w.replyErr(err)
		}
	}
	return
}
