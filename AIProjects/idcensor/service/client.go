package idcensor

import (
	"crypto/hmac"
	"crypto/sha1"
	"encoding/base64"
	"errors"
	"io"
	"io/ioutil"
	"net/http"
	"time"

	"github.com/qiniu/bytes/seekable"
)

type Client struct {
	access *AccessKey
}

func New(access *AccessKey) *Client {
	return &Client{
		access: access,
	}
}

func (client *Client) Send(method, absurl, contentType string, timeout time.Duration, body io.Reader) (data []byte, err error) {
	data, _, err = client.SendWithReqID(method, absurl, contentType, timeout, body)
	return
}

func (client *Client) SendWithReqID(method, absurl, contentType string, timeout time.Duration, body io.Reader) (data []byte, xReqId string, err error) {
	request, err := http.NewRequest(method, absurl, body)
	if err != nil {
		return
	}
	request.Header.Set("Content-Type", contentType)

	authClient := &http.Client{
		Transport: client,
		Timeout:   timeout,
	}

	response, err := authClient.Do(request)
	if err != nil {
		return
	}
	defer response.Body.Close()

	xReqId = response.Header.Get("X-Reqid")

	data, err = ioutil.ReadAll(response.Body)
	if response.StatusCode/100 != 2 {
		err = errors.New(string(data))
		data = nil
	}

	return
}

func (client *Client) RoundTrip(req *http.Request) (resp *http.Response, err error) {
	err = client.AuthRequest(req)
	if err != nil {
		return
	}

	return http.DefaultTransport.RoundTrip(req)
}

func (client *Client) AuthRequest(req *http.Request) (err error) {
	hash := hmac.New(sha1.New, client.access.Secret)

	urlobj := req.URL
	data := req.Method + " " + urlobj.Path
	if urlobj.RawQuery != "" {
		data += "?" + urlobj.RawQuery
	}
	io.WriteString(hash, data+"\nHost: "+req.Host)

	ctype := req.Header.Get("Content-Type")
	if ctype != "" {
		io.WriteString(hash, "\nContent-Type: "+ctype)
	}

	io.WriteString(hash, "\n\n")

	switch {
	case req.ContentLength != 0 && req.Body != nil &&
		ctype != "" && ctype != "application/octet-stream":
		seeker, tmperr := seekable.New(req)
		if tmperr != nil {
			err = tmperr
			return
		}

		hash.Write(seeker.Bytes())
	}

	sign := base64.URLEncoding.EncodeToString(hash.Sum(nil))

	req.Header.Set("Authorization", "Qiniu "+client.access.ID+":"+sign)

	return nil
}
