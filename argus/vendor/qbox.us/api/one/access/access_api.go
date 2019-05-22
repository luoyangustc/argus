package access

import (
	"net/http"
	"net/url"
	"strconv"
	"time"

	"qbox.us/servend/proxy_auth"

	"github.com/qiniu/rpc.v1"
	"github.com/qiniu/rpc.v1/lb.v2.1"
)

const (
	StatusNotFound       = 612
	StatusTooManyKeys    = 700
	StatusDisableAllKeys = 403
)

const (
	// state为0表示可用， state为1表示禁用
	StateEnabledKey  = 0
	StateDisabledKey = 1
)

type AppInfo struct {
	Uid     uint32 `json:"uid" bson:"uid"`
	AppName string `json:"appName" bson:"appName"`
	AppId   uint32 `json:"appId" bson:"appId"`

	Key          string    `json:"key,omitempty" bson:"key,omitempty"`
	Secret       string    `json:"secret,omitempty" bson:"secret,omitempty"`
	State        uint16    `json:"state,omitempty" bson:"state,omitempty"`
	LastModified time.Time `json:"last-modified,omitempty" bson:"last-modified,omitempty"`
	CreationTime time.Time `json:"creation-time,omitempty" bson:"creation-time,omitempty"`
	Comment      string    `json:"comment,omitempty" bson:"comment,omitempty"`

	Key2          string    `json:"key2,omitempty" bson:"key2,omitempty"`
	Secret2       string    `json:"secret2,omitempty" bson:"secret2,omitempty"`
	State2        uint16    `json:"state2,omitempty" bson:"state2,omitempty"`
	LastModified2 time.Time `json:"last-modified2,omitempty" bson:"last-modified2,omitempty"`
	CreationTime2 time.Time `json:"creation-time2,omitempty" bson:"creation-time2,omitempty"`
	Comment2      string    `json:"comment2,omitempty" bson:"comment2,omitempty"`
}

type AccessInfo struct {
	Key    string `json:"key"`
	Secret string `json:"secret"`
}

// =============================================================

type Client struct {
	Host string // 功能废弃，兼容保留
	Conn *lb.Client
}

func New(host string, t *proxy_auth.Transport) *Client {
	var rt http.RoundTripper
	if t != nil {
		rt = t
	}
	cfg := &lb.Config{
		Hosts:    []string{host},
		TryTimes: 1,
	}
	client := lb.New(cfg, rt)
	return &Client{host, client}
}

func NewWithMultiHosts(hosts []string, t *proxy_auth.Transport) *Client {
	var rt http.RoundTripper
	if t != nil {
		rt = t
	}
	cfg := &lb.Config{
		Hosts:    hosts,
		TryTimes: uint32(len(hosts)),
	}
	client := lb.New(cfg, rt)
	return &Client{Conn: client}
}

func (r *Client) AppInfo(l rpc.Logger, app string) (info AppInfo, err error) {

	params := map[string][]string{
		"app": {app},
	}
	err = r.Conn.CallWithForm(l, &info, "/appInfo", params)
	return
}

func (r *Client) NewAccess(l rpc.Logger, app string) (info AccessInfo, err error) {

	params := map[string][]string{
		"app": {app},
	}
	err = r.Conn.CallWithForm(l, &info, "/newAccess", params)
	return
}

func (r *Client) SetKeyState(l rpc.Logger, app string, accessKey string, state int) (err error) {

	params := map[string][]string{
		"app":   {app},
		"key":   {accessKey},
		"state": {strconv.Itoa(state)},
	}
	err = r.Conn.CallWithForm(l, nil, "/setKeyState", params)
	return
}

func (r *Client) DeleteAccess(l rpc.Logger, app string, accessKey string) (err error) {

	params := map[string][]string{
		"app": {app},
		"key": {accessKey},
	}
	err = r.Conn.CallWithForm(l, nil, "/deleteAccess", params)
	return
}

func (r *Client) GetByAccess(l rpc.Logger, accessKey string) (info AppInfo, err error) {

	params := map[string][]string{
		"key": {accessKey},
	}
	err = r.Conn.CallWithForm(l, &info, "/getByAccess", params)
	return
}

func (r *Client) SetComment(l rpc.Logger, app string, accessKey string, comment string) (err error) {
	params := map[string][]string{
		"app":     {app},
		"key":     {accessKey},
		"comment": {comment},
	}
	err = r.Conn.CallWithForm(l, nil, "/setComment", params)
	return
}

// 以下为兼容性代码，未来可能会删除
// ========================================================================

type OldAppInfo struct {
	Uid     uint32 `json:"uid"`
	AppName string `json:"appName"`
	Key     string `json:"key,omitempty"`
	Secret  string `json:"secret,omitempty"`
	Key2    string `json:"key2,omitempty"`
	Secret2 string `json:"secret2,omitempty"`
	AppId   uint32 `json:"appId"`
	State   uint16 `json:"state,omitempty"`  // 第1对 Key/Secret 的状态
	State2  uint16 `json:"state2,omitempty"` // 第2对 Key2/Secret2 的状态
}

func (r *Client) SyncAppInfo(l rpc.Logger, oldInfo OldAppInfo) (err error) {

	params := url.Values{}
	params.Add("uid", strconv.FormatUint(uint64(oldInfo.Uid), 10))
	params.Add("appName", oldInfo.AppName)
	params.Add("key", oldInfo.Key)
	params.Add("secret", oldInfo.Secret)
	params.Add("key2", oldInfo.Key2)
	params.Add("secret2", oldInfo.Secret2)
	params.Add("appId", strconv.FormatUint(uint64(oldInfo.AppId), 10))
	params.Add("state", strconv.FormatUint(uint64(oldInfo.State), 10))
	params.Add("state2", strconv.FormatUint(uint64(oldInfo.State2), 10))

	err = r.Conn.CallWithForm(l, nil, "/syncAppInfo", params)
	return
}
