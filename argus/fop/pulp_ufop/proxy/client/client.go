package proxy_client

import (
	"net/url"
)

type Client interface {
	PostForm(string, url.Values) ([]byte, error)
	Get(string, url.Values) ([]byte, error)
	Set(uid, utype uint32)
}
