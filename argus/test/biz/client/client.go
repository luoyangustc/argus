package client

import (
	"encoding/json"
	"fmt"

	"qiniu.com/argus/test/lib/auth"
	"qiniu.com/argus/test/lib/qnhttp"
)

type Client interface {
	Get(path string) (*qnhttp.Response, error)
	GetT(path string, ctype string) (*qnhttp.Response, error)
	GetB(path string, data interface{}) (*qnhttp.Response, error)
	PostWithJson(path string, data interface{}) (*qnhttp.Response, error)
	PostWithT(path string, ctype string, data interface{}) (*qnhttp.Response, error)
}

////////////////////////////////////////////////////////////////////////////////

var _ Client = _QiniuClient{}

type _QiniuClient struct {
	Host string
	User auth.AccessInfo
}

//NewQiniuClient 生成client
func NewQiniuClient(host string, user auth.AccessInfo) _QiniuClient {
	return _QiniuClient{Host: host, User: user}
}

//PostWithJson post请求
func (c _QiniuClient) PostWithJson(path string, data interface{}) (resp *qnhttp.Response, err error) {
	url := c.Host + path

	s := qnhttp.New()
	s.Header.Set("Content-Type", "application/json")
	{
		var bodyStr string
		if data != nil {
			bodyByte, _ := json.Marshal(data)
			bodyStr = string(bodyByte)
		}
		args := auth.QiniuAuthArgs{
			User:        c.User,
			Method:      "POST",
			URL:         url,
			Body:        bodyStr,
			ContentType: "application/json",
		}
		s.Header.Set("Authorization", "Qiniu "+auth.SignQiniuAuthToken(args))
	}
	// s.Header.Set("Host", c.Host)

	return s.Post(url, data, nil, nil)
}

//Get get
func (c _QiniuClient) Get(path string) (resp *qnhttp.Response, err error) {
	url := c.Host + path

	s := qnhttp.New()
	s.Header.Set("Content-Type", "application/json")
	{
		args := auth.QiniuAuthArgs{
			User:        c.User,
			Method:      "GET",
			URL:         url,
			Body:        "",
			ContentType: "application/json",
		}
		s.Header.Set("Authorization", "Qiniu "+auth.SignQiniuAuthToken(args))
	}
	// s.Header.Set("Host", c.Host)

	return s.Get(url, nil, nil, nil)
}

// GetB ... sends a GET request with body
func (c _QiniuClient) GetB(path string, data interface{}) (resp *qnhttp.Response, err error) {
	url := c.Host + path

	s := qnhttp.New()
	s.Header.Set("Content-Type", "application/json")
	{
		var bodyStr string
		if data != nil {
			bodyByte, _ := json.Marshal(data)
			bodyStr = string(bodyByte)
		}
		args := auth.QiniuAuthArgs{
			User:        c.User,
			Method:      "GET",
			URL:         url,
			Body:        bodyStr,
			ContentType: "application/json",
		}
		s.Header.Set("Authorization", "Qiniu "+auth.SignQiniuAuthToken(args))
	}
	return s.GetB(url, data, nil, nil)
}

//Get getT
func (c _QiniuClient) GetT(path string, ctype string) (resp *qnhttp.Response, err error) {
	url := c.Host + path

	s := qnhttp.New()
	if ctype != "" {
		s.Header.Set("Content-Type", ctype)
	}

	{
		s.Header.Set("Authorization", "QBox "+auth.SignQboxToken(c.User, url, ""))
	}
	// s.Header.Set("Host", c.Host)

	return s.Get(url, nil, nil, nil)
}

func (c _QiniuClient) PostWithT(path string, ctype string, data interface{}) (resp *qnhttp.Response, err error) {
	return
}

////////////////////////////////////////////////////////////////////////////////

var _ Client = _QiniuStubClient{}

type _QiniuStubClient struct {
	Host       string
	UID, Utype uint32
}

//NewQiniuStubClient 生成client
func NewQiniuStubClient(host string, uid, utype uint32) _QiniuStubClient {
	return _QiniuStubClient{Host: host, UID: uid, Utype: utype}
}

func (c _QiniuStubClient) PostWithJson(path string, data interface{}) (resp *qnhttp.Response, err error) {
	url := c.Host + path

	s := qnhttp.New()
	s.Header.Set("Content-Type", "application/json")
	s.Header.Set("Authorization", fmt.Sprintf("QiniuStub uid=%d&ut=%d", c.UID, c.Utype))
	//s.Header.Set("Host", c.Host)

	return s.Post(url, data, nil, nil)
}

//postWithFormData
func (c _QiniuStubClient) PostWithT(path string, ctype string, data interface{}) (resp *qnhttp.Response, err error) {
	url := c.Host + path
	fmt.Println(url)
	s := qnhttp.New()
	s.Header.Set("Content-Type", ctype)
	s.Header.Set("Authorization", fmt.Sprintf("QiniuStub uid=%d&ut=%d", c.UID, c.Utype))
	return s.Post(url, data, nil, nil)
}

//Get get
func (c _QiniuStubClient) Get(path string) (resp *qnhttp.Response, err error) {
	url := c.Host + path

	s := qnhttp.New()
	// s.Header.Del("Content-Type")
	s.Header.Set("Authorization", fmt.Sprintf("QiniuStub uid=%d&ut=%d", c.UID, c.Utype))
	// s.Header.Set("Host", c.Host)

	return s.Get(url, nil, nil, nil)
}

// GetB ...
func (c _QiniuStubClient) GetB(path string, data interface{}) (resp *qnhttp.Response, err error) {
	url := c.Host + path

	s := qnhttp.New()
	s.Header.Set("Content-Type", "application/json")
	s.Header.Set("Authorization", fmt.Sprintf("QiniuStub uid=%d&ut=%d", c.UID, c.Utype))
	return s.GetB(url, data, nil, nil)
}

//Get getT
func (c _QiniuStubClient) GetT(path string, ctype string) (resp *qnhttp.Response, err error) {

	return
}

////////////////////////////////////////////////////////////////////////////////

var _ Client = _QiniuStubClient{}

type _HTTPClient struct {
	Host       string
	UID, Utype uint32
}

//NewQiniuStubClient 生成client
func NewHTTPClient(host string) _HTTPClient {
	return _HTTPClient{Host: host}
}

func (c _HTTPClient) PostWithJson(path string, data interface{}) (resp *qnhttp.Response, err error) {
	url := c.Host + path

	s := qnhttp.New()
	s.Header.Set("Content-Type", "application/json")
	s.Header.Set("Host", c.Host)

	return s.Post(url, data, nil, nil)
}

//Get get
func (c _HTTPClient) Get(path string) (resp *qnhttp.Response, err error) {
	url := c.Host + path

	s := qnhttp.New()
	s.Header.Set("Host", c.Host)
	// s.Header.Del("Content-Type")

	return s.Get(url, nil, nil, nil)
}
