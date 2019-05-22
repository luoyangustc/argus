package qnhttp

/*
This module provides a Session object to manage and persist settings across
requests (cookies, auth, proxies).
*/

import (
	"bytes"
	"encoding/base64"
	"encoding/json"
	"errors"
	"io/ioutil"
	"log"
	"net"
	"net/http"
	"net/url"
	"strconv"
	"strings"
	"time"

	"io"
	"mime/multipart"
	"os"
	"path/filepath"
	// "github.com/onsi/ginkgo"
)

type Session struct {
	Client *http.Client
	Log    bool // Log request and response

	// Optional
	Userinfo *url.Userinfo

	// Optional defaults - can be overridden in a Request
	Header *http.Header
	Params *url.Values

	// Method specifies the HTTP method (GET, POST, PUT, etc.).
	// For client requests an empty string means GET.
	Method string

	// URL specifies either the URI being requested (for server
	// requests) or the URL to access (for client requests).
	//
	// For server requests the URL is parsed from the URI
	// supplied on the Request-Line as stored in RequestURI.  For
	// most requests, fields other than Path and RawQuery will be
	// empty. (See RFC 2616, Section 5.1.2)
	//
	// For client requests, the URL's Host specifies the server to
	// connect to, while the Request's Host field optionally
	// specifies the Host header value to send in the HTTP
	// request.
	URL string

	// Body is the request's body.
	Body string

	RetryTimeout int // 默认是 0，不开启超时重试
}

func isTimeoutErr(err error) bool {
	// go ver < 1.6 的 url.Error 没有实现 net.Error
	if uerr, ok := err.(*url.Error); ok {
		err = uerr.Err
	}
	if nerr, ok := err.(net.Error); ok {
		return nerr.Timeout()
	}
	return false
}

func (s *Session) Send(r *Request) (response *Response, err error) {
	for i := 0; i < s.RetryTimeout+1; i++ {
		response, err = s.send(r)
		if response.status == 504 || response.status == 424 {
			s.log("=============Response code:" + strconv.Itoa(response.status) + "，Retry One time =============")
			response, err = s.send(r)
		}
		if err != nil && s.RetryTimeout > 0 && isTimeoutErr(err) {
			s.log("Request Timeout, try again ", err)
			continue
		}
		break
	}
	return
}

// Send constructs and sends an HTTP request.
func (s *Session) send(r *Request) (response *Response, err error) {
	r.Method = strings.ToUpper(r.Method)
	//
	// Create a URL object from the raw url string.  This will allow us to compose
	// query parameters programmatically and be guaranteed of a well-formed URL.
	//
	u, err := url.Parse(r.Url)
	if err != nil {
		s.log("URL", r.Url)
		s.log(err)
		return
	}
	//
	// Default query parameters
	//
	p := url.Values{}
	if s.Params != nil {
		for k, v := range *s.Params {
			p[k] = v
		}
	}
	//
	// Parameters that were present in URL
	//
	if u.Query() != nil {
		for k, v := range u.Query() {
			p[k] = v
		}
	}
	//
	// User-supplied params override default
	//
	if r.Params != nil {
		for k, v := range *r.Params {
			p[k] = v
		}
	}
	//
	// Encode parameters
	// By JICHANGJUN, 2015-1-5
	// Encode 之后会破坏我们URL的结构，比如在做镜像回源时，我们的URL可以是这样：
	// http://80cyt7.com1.z0.glb.clouddn.com/Fy76txJL?qiniu_mirror=aHR0cDovLzE5Mi4xNjguMjAuMTAxOjgwOTAvY3VzdG9tSGVhZGVycw==&e=1451992706&token=0tf5awMVxwf8WrEvrjtbiZrdRZRJU-91JgCqTOC8:DQS-9PSp5io_3zcREHAPJE-xvhQ
	// 但是经过Encode之后:
	// http://80cyt7.com1.z0.glb.clouddn.com/Fy76txJL?e=1451992706&qiniu_mirror=aHR0cDovLzE5Mi4xNjguMjAuMTAxOjgwOTAvY3VzdG9tSGVhZGVycw%3D%3D&token=0tf5awMVxwf8WrEvrjtbiZrdRZRJU-91JgCqTOC8%3ADQS-9PSp5io_3zcREHAPJE-xvhQ
	// u.RawQuery = p.Encode()
	//
	// Attach params to response
	//
	r.Params = &p
	//
	// Create a Request object; if populated, Data field is JSON encoded as
	// request body
	//
	header := http.Header{}
	if s.Header != nil {
		for k, _ := range *s.Header {
			v := s.Header.Get(k)
			header.Set(k, v)
		}
	}
	var req *http.Request
	var buf *bytes.Buffer
	if r.Payload != nil {
		if r.RawPayload {
			var ok bool
			// buf can be nil interface at this point
			// so we'll do extra nil check
			buf, ok = r.Payload.(*bytes.Buffer)
			if !ok {
				err = errors.New("Payload must be of type *bytes.Buffer if RawPayload is set to true")
				return
			}
			req, err = http.NewRequest(r.Method, u.String(), buf)
		} else if s.Header.Get("Content-Type") == "multipart/form-data" {
			var params = r.Payload.(map[string]string)
			file, _ := os.Open(params["filepath"])
			body := &bytes.Buffer{}
			writer := multipart.NewWriter(body)
			part, err := writer.CreateFormFile("file", filepath.Base(params["filepath"]))
			if err != nil {
				return nil, err
			}
			_, _ = io.Copy(part, file)

			for key, val := range params {
				if key != "filepath" {
					_ = writer.WriteField(key, val)
				}
			}
			err = writer.Close()
			if err != nil {
				return nil, err
			}
			req, _ = http.NewRequest(r.Method, u.String(), body)
			header.Set("Content-Type", writer.FormDataContentType())

		} else if s.Header.Get("Content-Type") == "text/plain" {
			b := []byte(r.Payload.(string))
			buf = bytes.NewBuffer(b)
			req, err = http.NewRequest(r.Method, u.String(), buf)
		} else {
			var b []byte
			if s.Header.Get("Content-Type") == "application/x-www-form-urlencoded" ||
				s.Header.Get("Content-Type") == "application/octet-stream" {
				b = []byte(r.Payload.(string))
			} else {
				b, err = json.Marshal(&r.Payload)
				if err != nil {
					s.log(err)
					return
				}
			}
			buf = bytes.NewBuffer(b)
			if buf != nil {
				req, err = http.NewRequest(r.Method, u.String(), buf)
			} else {
				req, err = http.NewRequest(r.Method, u.String(), nil)
			}
		}
		if err != nil {
			s.log(err)
			return
		}
		// println(req.Header.Get("Content-Type"))
		// Overwrite the content type to json since we're pushing the payload as json
		// header.Set("Content-Type", "application/x-www-form-urlencoded")
	} else { // no data to encode
		req, err = http.NewRequest(r.Method, u.String(), nil)
		if err != nil {
			s.log(err)
			return
		}

	}
	//
	// Merge Session and Request options
	//
	var userinfo *url.Userinfo
	if u.User != nil {
		userinfo = u.User
	}
	if s.Userinfo != nil {
		userinfo = s.Userinfo
	}
	// Prefer Request's user credentials
	if r.Userinfo != nil {
		userinfo = r.Userinfo
	}
	if r.Header != nil {
		for k, v := range *r.Header {
			header.Set(k, v[0]) // Is there always guarnateed to be at least one value for a header?
		}
	}
	// if header.Get("Accept") == "" {
	// 	header.Add("Accept", "application/json") // Default, can be overridden with Opts
	// }

	if host := header.Get("Host"); host != "" {
		req.Host = host
	}

	req.Header = header
	//
	// Set HTTP Basic authentication if userinfo is supplied
	//
	if userinfo != nil {
		pwd, _ := userinfo.Password()
		req.SetBasicAuth(userinfo.Username(), pwd)
		if u.Scheme != "https" {
			s.log("WARNING: Using HTTP Basic Auth in cleartext is insecure.")
		}
	}
	//
	// Execute the HTTP request
	//

	// Debug log request
	s.log("--------------------------------------------------------------------------------")
	s.log("REQUEST")
	s.log("--------------------------------------------------------------------------------")
	s.log("Method:", req.Method)
	s.log("URL:", req.URL)
	s.log("Header:", req.Header)
	s.log("Form:", req.Form)
	s.log("Payload:")
	if r.RawPayload && s.Log && buf != nil {
		s.log(base64.StdEncoding.EncodeToString(buf.Bytes()))
	} else {
		str := pretty(r.Payload)
		if len(str) < 1024 {
			s.log(str)
		} else {
			s.log("Payload size was more then 1024 bytes, Not print")
		}
	}
	r.timestamp = time.Now()
	var client *http.Client
	if s.Client != nil {
		client = s.Client
	} else {
		client = &http.Client{}
		s.Client = client
	}
	resp, err := client.Do(req)
	if err != nil {
		s.log(err)
		return
	}
	defer resp.Body.Close()
	r.status = resp.StatusCode
	r.response = resp
	//
	// Unmarshal
	//
	r.body, err = ioutil.ReadAll(resp.Body)
	if err != nil {
		s.log("WARN: Failed to read repsonse body, with exception ", err)
	}
	if string(r.body) != "" {
		if resp.StatusCode < 300 && r.Result != nil {
			err = json.Unmarshal(r.body, r.Result)
		}
		if resp.StatusCode >= 400 && r.Error != nil {
			_ = json.Unmarshal(r.body, r.Error) // Should we ignore unmarshall error?
		}
	}
	if r.CaptureResponseBody {
		r.ResponseBody = bytes.NewBuffer(r.body)
	}
	rsp := Response(*r)
	response = &rsp

	// Debug log response
	s.log("--------------------------------------------------------------------------------")
	s.log("RESPONSE")
	s.log("--------------------------------------------------------------------------------")
	s.log("Status: ", response.status)
	s.log("Header:")
	s.log(response.HttpResponse().Header)
	s.log("Body:")

	//if response.body != nil && len(response.body) <= 512 {
	//	raw := json.RawMessage{}
	//	if json.Unmarshal(response.body, &raw) == nil {
	//		s.log(pretty(&raw))
	//	} else {
	//		s.log(pretty(response.RawText()))
	//	}
	//} else {
	//	s.log("Empty response body or body size was too large (>=512)")
	//}

	if response.body == nil {
		s.log("Empty response body")
	} else if len(response.body) <= 8192 {
		raw := json.RawMessage{}
		if json.Unmarshal(response.body, &raw) == nil {
			s.log(pretty(&raw))
		} else {
			s.log(pretty(response.RawText()))
		}
	} else {
		s.log("Response body size was too large (>=8192)")
	}

	return
}

// Get sends a GET request.
func (s *Session) Get(url string, p *url.Values, result, errMsg interface{}) (*Response, error) {
	r := Request{
		Method: "GET",
		Url:    url,
		Params: p,
		Result: result,
		Error:  errMsg,
	}
	return s.Send(&r)
}

// GetB ... sends a GET request with body
func (s *Session) GetB(url string, payload, result, errMsg interface{}) (*Response, error) {
	r := Request{
		Method:  "GET",
		Url:     url,
		Payload: payload,
		Result:  result,
		Error:   errMsg,
	}
	return s.Send(&r)
}

// Options sends an OPTIONS request.
func (s *Session) Options(url string, result, errMsg interface{}) (*Response, error) {
	r := Request{
		Method: "OPTIONS",
		Url:    url,
		Result: result,
		Error:  errMsg,
	}
	return s.Send(&r)
}

// Head sends a HEAD request.
func (s *Session) Head(url string, result, errMsg interface{}) (*Response, error) {
	r := Request{
		Method: "HEAD",
		Url:    url,
		Result: result,
		Error:  errMsg,
	}
	return s.Send(&r)
}

// Post sends a POST request.
func (s *Session) Post(url string, payload, result, errMsg interface{}) (*Response, error) {
	r := Request{
		Method:  "POST",
		Url:     url,
		Payload: payload,
		Result:  result,
		Error:   errMsg,
	}
	return s.Send(&r)
}

// Put sends a PUT request.
func (s *Session) Put(url string, payload, result, errMsg interface{}) (*Response, error) {
	r := Request{
		Method:  "PUT",
		Url:     url,
		Payload: payload,
		Result:  result,
		Error:   errMsg,
	}
	return s.Send(&r)
}

// Patch sends a PATCH request.
func (s *Session) Patch(url string, payload, result, errMsg interface{}) (*Response, error) {
	r := Request{
		Method:  "PATCH",
		Url:     url,
		Payload: payload,
		Result:  result,
		Error:   errMsg,
	}
	return s.Send(&r)
}

// Delete sends a DELETE request.
func (s *Session) Delete(url string, payload, result, errMsg interface{}) (*Response, error) {
	r := Request{
		Method:  "DELETE",
		Url:     url,
		Payload: payload,
		Result:  result,
		Error:   errMsg,
	}
	return s.Send(&r)
}

// Debug method for logging
// Centralizing logging in one method
// avoids spreading conditionals everywhere
func (s *Session) log(args ...interface{}) {
	if s.Log {
		// log.SetOutput(ginkgo.GinkgoWriter)
		log.Println(args...)
	}
}
