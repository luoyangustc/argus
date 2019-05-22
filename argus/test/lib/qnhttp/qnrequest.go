package qnhttp

import (
	"net/http"
	"os"
	"strconv"
)

func New() *Session {
	s := Session{RetryTimeout: 2} // 超时重试 2 次
	s.Header = &http.Header{}
	s.Header.Add("User-Agent", "qiniu_qa")
	s.Header.Add("Content-Type", "application/x-www-form-urlencoded")
	s.Log, _ = strconv.ParseBool(os.Getenv("DEBUG"))
	return &s
}

func (s *Session) Set(header, value string) *Session {
	s.Header.Set(header, value)
	return s
}

func (s *Session) Del(header string) *Session {
	s.Header.Del(header)
	return s
}

func (s *Session) SetClient(client *http.Client) *Session {
	s.Client = client
	return s
}

func (s *Session) SetTransport(transport *http.Transport) *Session {
	s.Client = &http.Client{Transport: transport}
	return s
}

func (s *Session) DisableKeepAlive(isDisable bool) *Session {
	tr := &http.Transport{DisableKeepAlives: isDisable}
	s.Client = &http.Client{Transport: tr}
	return s
}
