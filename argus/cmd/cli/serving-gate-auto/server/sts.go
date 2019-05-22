package server

import (
	"encoding/base64"
	"fmt"
	"io"
	"io/ioutil"
	"net/http"
	"net/url"
	"strings"

	"github.com/pkg/errors"
)

func (s *server) genStsFetchUri(uri string) string {
	u := url.Values{}
	u.Set("uri", uri)
	stsUrl := s.cfg.StsHost + "/v1/fetch?" + u.Encode()
	return stsUrl
}

func (s *server) readUrlBody(url string) (body []byte, err error) {
	if s.mock {
		return []byte(fmt.Sprintf("body(%s)", url)), nil
	}
	resp, err := http.Get(url)
	if err != nil {
		return nil, errors.Wrapf(err, "http.Get %v", url)
	}
	defer resp.Body.Close()
	if resp.StatusCode/100 != 2 {
		_, _ = io.Copy(ioutil.Discard, resp.Body)
		return nil, errors.Errorf("call %v, status %v", url, resp.StatusCode)
	}
	bodyStream := io.LimitReader(resp.Body, MaxBodySize)
	bodyBuf, err := ioutil.ReadAll(bodyStream)
	if err != nil {
		_, _ = io.Copy(ioutil.Discard, resp.Body)
		return nil, errors.Wrapf(err, "read body %v", url)
	}
	return bodyBuf, nil
}

func (s *server) bufToUri(buf []byte) string {
	return dataURIPrefix + base64.StdEncoding.EncodeToString(buf)
}

func (s *server) fixUri(uri string) (string, error) {
	switch {
	case strings.HasPrefix(uri, "sts://"):
		u := strings.Replace(uri, "sts://", "http://", 1)
		buf, err := s.readUrlBody(u)
		if err != nil {
			return "", errors.Wrap(err, "readUrlBody")
		}
		return s.bufToUri(buf), nil
	case strings.HasPrefix(uri, "qiniu://"), strings.HasPrefix(uri, "http://10.200.30.13:10000"):
		u := s.genStsFetchUri(uri)
		buf, err := s.readUrlBody(u)
		if err != nil {
			return "", errors.Wrap(err, "readUrlBody")
		}
		return s.bufToUri(buf), nil
	default:
		return uri, nil
	}
}
