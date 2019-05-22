package jsonlog

import (
	"bytes"
	"encoding/json"
	"io"
	"net/http"
	"strconv"
	"strings"
	"time"

	"github.com/qiniu/bytes/seekable"
	"github.com/qiniu/errors"
	"github.com/qiniu/http/httputil.v1"
	"github.com/qiniu/http/servestk.v1"

	qbytes "github.com/qiniu/bytes"
	. "github.com/qiniu/http/audit/proto"
)

// ----------------------------------------------------------

type responseWriter struct {
	http.ResponseWriter
	body      *qbytes.Writer
	extra     M
	written   int64
	startT    int64
	mod       string
	code      int
	xlog      bool
	skip      bool
	noLogBody bool // Affect response with 2xx only.
}

var xlogKey = "X-Log"

// 注意：此函数线程不安全，只用于初始化 xlogKey 变量。
func SetXlogKey(key string) {
	xlogKey = key
}

const xwanKey = "X-Warn"
const maxXlogLen = 509 // 512 - len("...")

func (r *responseWriter) Write(buf []byte) (n int, err error) {
	if r.xlog {
		r.logDuration(r.code)
		fullXlog, trunced := r.xlogMerge()
		if trunced {
			defer func() {
				r.setXlog(fullXlog)
			}()
		}
		r.xlog = false
		if r.code/100 == 2 && r.noLogBody {
			r.skip = true
		}
	}
	n, err = r.ResponseWriter.Write(buf)
	r.written += int64(n)
	if n == len(buf) && !r.skip {
		n2, _ := r.body.Write(buf)
		if n2 == n {
			return
		}
	}
	r.skip = true
	return
}

func (r *responseWriter) getBody() []byte {
	if r.skip {
		return nil
	}
	return r.body.Bytes()
}

func (r *responseWriter) ExtraDisableBodyLog() {
	r.noLogBody = true
}

func (r *responseWriter) xlogMerge() (fullXlog string, trunc bool) {
	headers := r.Header()
	v, ok := headers[xlogKey]
	if !ok {
		return
	}

	defer func() {
		if len(fullXlog) > maxXlogLen {
			trunc = true
			headers[xlogKey] = []string{"..." + fullXlog[len(fullXlog)-maxXlogLen:]}
		}
	}()

	if len(v) <= 1 {
		fullXlog = v[0]
		return
	}
	fullXlog = strings.Join(v, ";")
	headers[xlogKey] = []string{fullXlog}
	return
}

func (r *responseWriter) setXlog(xlog string) {
	headers := r.Header()
	_, ok := headers[xlogKey]
	if !ok {
		return
	}
	headers[xlogKey] = []string{xlog}
}

//
// X-Log: xxx; MOD[:duration][/code]
//
func (r *responseWriter) WriteHeader(code int) {
	if r.xlog {
		r.logDuration(code)
		fullXlog, trunced := r.xlogMerge()
		if trunced {
			defer func() {
				r.setXlog(fullXlog)
			}()
		}
		r.xlog = false
		if r.code/100 == 2 && r.noLogBody {
			r.skip = true
		}
	}
	r.ResponseWriter.WriteHeader(code)
	r.code = code
}

func (r *responseWriter) ExtraWrite(key string, val interface{}) {
	if r.extra == nil {
		r.extra = make(M)
	}
	r.extra[key] = val
}

func (r *responseWriter) ExtraAddInt64(key string, val int64) {
	if r.extra == nil {
		r.extra = make(M)
	}
	if v, ok := r.extra[key]; ok {
		val += v.(int64)
	}
	r.extra[key] = val
}

func (r *responseWriter) ExtraAddString(key string, val string) {
	if r.extra == nil {
		r.extra = make(M)
	}
	var v []string
	if v1, ok := r.extra[key]; ok {
		v = v1.([]string)
	}
	r.extra[key] = append(v, val)
}

func (r *responseWriter) GetStatusCode() int {
	return r.code
}

func (r *responseWriter) logDuration(code int) {
	h := r.Header()
	dur := (time.Now().UnixNano() - r.startT) / 1e6
	xlog := r.mod
	if dur > 0 {
		xlog += ":" + strconv.FormatInt(dur, 10)
	}
	if code/100 != 2 {
		xlog += "/" + strconv.Itoa(code)
	}
	h[xlogKey] = append(h[xlogKey], xlog)
}

// ----------------------------------------------------------

func Info(w http.ResponseWriter, key string, val interface{}) {
	ew, ok := w.(extraWriter)
	if !ok {
		ew, ok = getExtraWriter(w)
	}
	if ok {
		ew.ExtraWrite(key, val)
	}
}

func AddInt64(w http.ResponseWriter, key string, val int64) {
	ew, ok := w.(extraInt64Adder)
	if !ok {
		ew, ok = getExtraInt64Adder(w)
	}
	if ok {
		ew.ExtraAddInt64(key, val)
	}
}

func Xwarn(w http.ResponseWriter, val string) {
	ew, ok := w.(extraStringAdder)
	if !ok {
		ew, ok = getExtraStringAdder(w)
	}
	if ok {
		ew.ExtraAddString(xwanKey, val)
	}
}

func DisableBodyLog(w http.ResponseWriter) {
	w1, ok := w.(extraBodyLogDisabler)
	if !ok {
		w1, ok = getExtraBodyLogDisabler(w)
	}
	if ok {
		w1.ExtraDisableBodyLog()
	}
}

// ----------------------------------------------------------

type LogWriter interface {
	Log(msg []byte) error
}

type Decoder interface {
	DecodeRequest(req *http.Request) (url_ string, header, params M)
	DecodeResponse(header http.Header, bodyThumb []byte, extra, params M) (resph M, body []byte)
}

type BaseDecoder struct {
}

func set(h M, header http.Header, key string) {
	if v, ok := header[key]; ok {
		h[key] = v[0]
	}
}

func ip(addr string) string {
	pos := strings.Index(addr, ":")
	if pos < 0 {
		return addr
	}
	return addr[:pos]
}

func queryToJson(m map[string][]string) (h M, err error) {

	h = make(M)
	for k, v := range m {
		if len(v) == 1 {
			h[k] = v[0]
		} else {
			h[k] = v
		}
	}
	return
}

func (r BaseDecoder) DecodeRequest(req *http.Request) (url_ string, h, params M) {

	h = M{"IP": ip(req.RemoteAddr), "Host": req.Host}
	ct, ok := req.Header["Content-Type"]
	if ok {
		h["Content-Type"] = ct[0]
	}
	if req.URL.RawQuery != "" {
		h["RawQuery"] = req.URL.RawQuery
	}

	set(h, req.Header, "User-Agent")
	set(h, req.Header, "Range")
	set(h, req.Header, "Refer")
	set(h, req.Header, "Content-Length")
	set(h, req.Header, "If-None-Match")
	set(h, req.Header, "If-Modified-Since")
	set(h, req.Header, "X-Real-Ip")
	set(h, req.Header, "X-Forwarded-For")
	set(h, req.Header, "X-Scheme")
	set(h, req.Header, "X-Remote-Ip")
	set(h, req.Header, "X-Reqid")
	set(h, req.Header, "X-Id")
	set(h, req.Header, "X-From-Cdn")
	set(h, req.Header, "X-Tencent-Ua")
	set(h, req.Header, "X-From-Proxy-Getter")
	set(h, req.Header, "X-From-Fsrcproxy")
	set(h, req.Header, "Cdn-Src-Ip")
	set(h, req.Header, "Cdn-Scheme")
	set(h, req.Header, "X-Upload-Encoding")
	set(h, req.Header, "Accept-Encoding")

	// 记录非七牛 CDN 客户请求来源 CDN，方便排查问题
	// 网宿: wangsu, 蓝汛: ChinaCache, 帝联: dnion, 浩瀚: Power-By-HaoHan, 华云: 51CDN, 同兴: TXCDN
	set(h, req.Header, "Cdn")

	url_ = req.URL.Path
	if ok {
		switch ct[0] {
		case "application/x-www-form-urlencoded":
			seekable, err := seekable.New(req)
			if err == nil {
				req.ParseForm()
				params, _ = queryToJson(req.Form)
				seekable.SeekToBegin()
			}
		}
	}
	if params == nil {
		params = make(M)
	}
	return
}

func (r BaseDecoder) DecodeResponse(header http.Header, bodyThumb []byte, h, params M) (resph M, body []byte) {

	if h == nil {
		h = make(M)
	}

	ct, ok := header["Content-Type"]
	if ok {
		h["Content-Type"] = ct[0]
	}
	if xlog, ok := header["X-Log"]; ok {
		h["X-Log"] = xlog
	}
	if xmlog, ok := header["X-M-Log"]; ok {
		h["X-M-Log"] = xmlog
	}
	set(h, header, "X-Reqid")
	set(h, header, "X-M-Reqid")
	set(h, header, "X-Id")
	set(h, header, "X-Qnm-Cache")
	set(h, header, "Content-Length")
	set(h, header, "Content-Encoding")
	set(h, header, "X-Qiniu-Root-Cause")

	if ok && ct[0] == "application/json" && header.Get("Content-Encoding") != "gzip" {
		if -1 == bytes.IndexAny(bodyThumb, "\n\r") {
			body = bodyThumb
		}
	}
	resph = h
	return
}

var DefaultDecoder BaseDecoder

// ----------------------------------------------------------

type Logger struct {
	w     LogWriter
	dec   Decoder
	event Event
	mod   string
	limit int
	xlog  bool
}

func New(mod string, w LogWriter, dec Decoder, limit int) *Logger {
	if dec == nil {
		dec = DefaultDecoder
	}
	return &Logger{w, dec, nil, mod, limit, true}
}

func NewEx(mod string, w LogWriter, dec Decoder, limit int, xlog bool) *Logger {
	if dec == nil {
		dec = DefaultDecoder
	}
	return &Logger{w, dec, nil, mod, limit, xlog}
}

func (r *Logger) SetEvent(event Event) {

	r.event = event
}

type xBody struct {
	r io.ReadCloser
	n int
}

func (p *xBody) Read(d []byte) (n int, err error) {
	n, err = p.r.Read(d)
	p.n += n
	return
}

func (p *xBody) Close() error {
	return p.r.Close()
}

func newReadCloser(readCloser io.ReadCloser) *xBody {
	return &xBody{r: readCloser}
}

func (r *Logger) Handler(
	w http.ResponseWriter, req *http.Request, f func(http.ResponseWriter, *http.Request)) {

	url_, headerM, paramsM := r.dec.DecodeRequest(req)
	if url_ == "" { // skip
		servestk.SafeHandler(w, req, f)
		return
	}
	contentLength := int64(-1)
	if cl := headerM["Content-Length"]; cl != nil {
		contentLength, _ = strconv.ParseInt(cl.(string), 10, 64)
	}
	var xbody *xBody
	if contentLength < 0 {
		xbody = newReadCloser(req.Body)
		req.Body = xbody
	}

	var header, params, resph []byte
	if len(paramsM) != 0 {
		params, _ = json.Marshal(paramsM)
		if len(params) > 4096 {
			params, _ = json.Marshal(M{"discarded": len(params)})
		}
	}

	body := qbytes.NewWriter(make([]byte, r.limit))
	b := bytes.NewBuffer(nil)
	startTime := time.Now().UnixNano()
	w1 := &responseWriter{
		ResponseWriter: w,
		body:           body,
		code:           200,
		xlog:           r.xlog,
		mod:            r.mod,
		startT:         startTime,
	}

	event := r.event
	if event == nil {
		servestk.SafeHandler(w1, req, f)
	} else {
		req1 := &Request{
			StartTime: startTime,
			Method:    req.Method,
			Mod:       r.mod,
			Path:      url_,
			Header:    headerM,
			Params:    paramsM,
		}
		id, err := event.OnStartReq(req1)
		if err != nil {
			httputil.Error(w1, err)
		} else {
			servestk.SafeHandler(w1, req, f)
			event.OnEndReq(id)
		}
	}

	startTime /= 100
	endTime := time.Now().UnixNano() / 100

	if xbody != nil {
		headerM["bs"] = xbody.n
	}
	if len(headerM) != 0 {
		header, _ = json.Marshal(headerM)
	}

	b.WriteString("REQ\t")
	b.WriteString(r.mod)
	b.WriteByte('\t')

	b.WriteString(strconv.FormatInt(startTime, 10))
	b.WriteByte('\t')
	b.WriteString(req.Method)
	b.WriteByte('\t')
	b.WriteString(url_)
	b.WriteByte('\t')
	b.Write(header)
	b.WriteByte('\t')
	b.Write(params)
	b.WriteByte('\t')

	resphM, respb := r.dec.DecodeResponse(w1.Header(), w1.getBody(), w1.extra, paramsM)
	if len(resphM) != 0 {
		resph, _ = json.Marshal(resphM)
	}

	b.WriteString(strconv.Itoa(w1.code))
	b.WriteByte('\t')
	b.Write(resph)
	b.WriteByte('\t')
	b.Write(respb)
	b.WriteByte('\t')
	b.WriteString(strconv.FormatInt(w1.written, 10))
	b.WriteByte('\t')
	b.WriteString(strconv.FormatInt(endTime-startTime, 10))

	err := r.w.Log(b.Bytes())
	if err != nil {
		errors.Info(err, "jsonlog.Handler: Log failed").Detail(err).Warn()
	}
}

// ----------------------------------------------------------
