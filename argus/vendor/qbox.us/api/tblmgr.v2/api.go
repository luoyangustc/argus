package tblmgr

import (
	"encoding/base64"
	"fmt"
	"net/http"
	"strconv"
	"strings"

	"github.com/qiniu/http/httputil.v1"

	"github.com/qiniu/rpc.v1"
	"github.com/qiniu/rpc.v1/lb.v2.1"
)

var ErrInvalidBucket = httputil.NewError(400, "invalid bucket name")

type BucketEntry struct {
	Tbl        string      `bson:"tbl" json:"tbl"`
	Itbl       uint32      `bson:"itbl" json:"itbl"`
	PhyTbl     string      `bson:"phy" json:"phy"`
	Uid        uint32      `bson:"uid" json:"uid"`
	Zone       string      `bson:"zone" json:"zone"`
	Region     string      `bson:"region" json:"region"`
	Global     bool        `bson:"global" json:"global"`
	Line       bool        `bson:"line" json:"line"`
	Ctime      int64       `bson:"ctime" json:"ctime"`
	Oitbl      uint32      `bson:"oitbl" json:"oitbl"`
	Ouid       uint32      `bson:"ouid" json:"ouid"`
	Otbl       string      `bson:"otbl" json:"otbl"`
	Perm       uint32      `bson:"perm" json:"perm"`
	ShareUsers []ShareUser `bson:"share_users" json:"share_users"`
}

type ShareUser struct {
	Uid  uint32 `json:"uid"`
	Tbl  string `json:"tbl"`
	Perm uint32 `json:"perm"`
}

const (
	RD = 1
	RW = 2
)

// Must with authorization
type Client struct {
	Conn *lb.Client
}

func New(host string, t http.RoundTripper) *Client {
	cfg := &lb.Config{
		Hosts:    []string{host},
		TryTimes: 1,
	}
	client := lb.New(cfg, t)
	return &Client{client}
}

func NewWithMultiHosts(hosts []string, t http.RoundTripper) *Client {
	cfg := &lb.Config{
		Hosts:    hosts,
		TryTimes: uint32(len(hosts)),
	}
	client := lb.New(cfg, t)
	return &Client{client}
}

func (c Client) Share(l rpc.Logger, tbl string, uid_dest uint32, perm int32) (err error) {
	return c.ShareWithName(l, tbl, uid_dest, tbl, perm)
}

func (c Client) ShareWithName(l rpc.Logger, tbl string, uid_dest uint32, tbl_dest string, perm int32) (err error) {
	url := fmt.Sprintf("/share/%s/to/%d/perm/%d/name/%s", tbl, uid_dest, perm, tbl_dest)
	err = c.Conn.Call(l, nil, url)
	return
}

func (c Client) CancelShare(l rpc.Logger, tbl string, uid_dest uint32) (err error) {
	return c.Share(l, tbl, uid_dest, -1)
}

func (c *Client) Bucket(l rpc.Logger, tbl string) (entry BucketEntry, err error) {

	if strings.HasPrefix(tbl, "/") {
		// bucket其他字符按说也不应该出现/，已经有用户在创建bucket的时候包含/，为了兼容，这里只检查前缀
		err = ErrInvalidBucket
		return
	}
	err = c.Conn.Call(l, &entry, "/bucket/"+tbl)
	return
}

func (c Client) Buckets(l rpc.Logger, region string) (entrys []BucketEntry, err error) {
	return c.BucketsWithShared(l, region, false)
}

func (c Client) BucketsWithShared(l rpc.Logger, region string, shared bool) (entrys []BucketEntry, err error) {

	url := "/v2/buckets"
	params := make(map[string][]string)
	if region != "" {
		params["region"] = []string{region}
	}
	if shared {
		params["shared"] = []string{"true"}
	}
	if len(params) == 0 {
		err = c.Conn.Call(l, &entrys, url)
	} else {
		err = c.Conn.CallWithForm(l, &entrys, url, params)
	}
	return
}

func (c Client) GlbBuckets(l rpc.Logger, region, global string) (entrys []BucketEntry, err error) {
	return c.GlbBucketsWithShared(l, region, global, false)
}

func (c Client) GlbBucketsWithShared(l rpc.Logger, region, global string, shared bool) (entrys []BucketEntry, err error) {

	url := "/v2/buckets"
	params := make(map[string][]string)
	if region != "" {
		params["region"] = []string{region}
	}
	if global != "" {
		params["global"] = []string{global}
	}
	if shared {
		params["shared"] = []string{"true"}
	}

	if len(params) == 0 {
		err = c.Conn.Call(l, &entrys, url)
	} else {
		err = c.Conn.CallWithForm(l, &entrys, url, params)
	}
	return
}

func (c Client) Mkbucket(l rpc.Logger, tbl, region string) error {

	if strings.HasPrefix(tbl, "/") {
		// bucket以/开头会导致301，这里直接直接报错返回400
		return ErrInvalidBucket
	}
	url := "/mkbucket/" + tbl
	if region != "" {
		url += "/region/" + region
	}
	return c.Conn.Call(l, nil, url)
}

func (c Client) MkbucketV2(l rpc.Logger, tbl, region string) error {

	url := "/mkbucketv2/" + base64.URLEncoding.EncodeToString([]byte(tbl))
	if region != "" {
		url += "/region/" + region
	}
	return c.Conn.Call(l, nil, url)
}

func (c Client) GlbMkbucket(l rpc.Logger, tbl, region string, global bool) error {

	if strings.HasPrefix(tbl, "/") {
		return ErrInvalidBucket
	}
	url := "/mkbucket/" + tbl
	if global {
		url += "/global/true"
	}
	if region != "" {
		url += "/region/" + region
	}
	return c.Conn.Call(l, nil, url)
}

func (c Client) GlbMkbucketV2(l rpc.Logger, tbl, region string, global bool) error {

	url := "/mkbucketv2/" + base64.URLEncoding.EncodeToString([]byte(tbl))
	if global {
		url += "/global/true"
	}
	if region != "" {
		url += "/region/" + region
	}
	return c.Conn.Call(l, nil, url)
}

func (c Client) SetGlobal(l rpc.Logger, uid uint32, tbl string) error {
	url := "/admin/setglobal"
	params := map[string][]string{"uid": {strconv.FormatUint(uint64(uid), 10)}, "tbl": {tbl}}
	return c.Conn.CallWithForm(l, nil, url, params)
}

// Shall without authorization
type ClientNullAuth struct {
	Conn *lb.Client
}

func NewNullAuth(host string, t http.RoundTripper) *ClientNullAuth {
	cfg := &lb.Config{
		Hosts:    []string{host},
		TryTimes: 1,
	}
	client := lb.New(cfg, t)
	return &ClientNullAuth{client}
}

func NewNullAuthWithMultiHosts(hosts []string, t http.RoundTripper) *ClientNullAuth {
	cfg := &lb.Config{
		Hosts:    hosts,
		TryTimes: uint32(len(hosts)),
	}
	client := lb.New(cfg, t)
	return &ClientNullAuth{client}
}

func (c ClientNullAuth) GetByItbl(l rpc.Logger, itbl uint32) (entry BucketEntry, err error) {
	err = c.Conn.Call(l, &entry, fmt.Sprintf("/itblbucket/%d", itbl))
	return
}
