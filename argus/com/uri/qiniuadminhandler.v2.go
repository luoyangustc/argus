package uri

import (
	"container/list"
	"context"
	"encoding/hex"
	"fmt"
	"net/http"
	"net/url"
	"strconv"
	"sync"
	"time"

	"github.com/pkg/errors"
	"github.com/qiniu/rpc.v3"
	"github.com/qiniu/xlog.v1"
	"qbox.us/qconf/qconfapi"
	"qiniu.com/argus/argus/com/auth"
	"qiniu.com/auth/qboxmac.v1"
	"qiniupkg.com/api.v7/kodo"
)

func WithAdminAkSkV2(conf QiniuAdminHandlerConfig, tr http.RoundTripper) Handler {
	handler := &qiniuAdminHandlerV2{
		QiniuAdminHandlerConfig: conf,
		qCli: qconfapi.New(&conf.Qconf),
	}
	if tr != nil {
		handler.Client = &http.Client{Transport: tr}
	}
	if conf.LC == nil {
		return handler
	}
	handler.lc = newCache(*conf.LC,
		func(ctx context.Context, k0 _Key) (interface{}, error) {
			k := k0.(_K)

			info := struct {
				Zone string `json:"zone"`
			}{}
			err := rpc.NewClientWithTransport(
				qboxmac.NewTransport(&qboxmac.Mac{AccessKey: k.AK, SecretKey: []byte(k.SK)}, nil)).
				Call(ctx, &info, "GET",
					fmt.Sprintf("http://%s/bucket/%s", handler.RSHost, k.Bucket),
				)
			if err != nil {
				return nil, err
			}

			zone := info.Zone
			ioHost := handler.IOHosts[zone]

			return _V{Zone: zone, IOHost: ioHost}, nil
		},
	)
	return handler
}

//----------------------------------------------------------------------------//

type QiniuAdminHandlerConfig struct {
	Qconf   qconfapi.Config   `json:"qconf"`
	LC      *CacheConfig      `json:"local_cache"`
	AdminAK string            `json:"admin_ak"`
	AdminSK string            `json:"admin_sk"`
	RSHost  string            `json:"rs_host"`
	IOHosts map[string]string `json:"io_hosts"`
}

type qiniuAdminHandlerV2 struct {
	QiniuAdminHandlerConfig
	qCli *qconfapi.Client
	*http.Client

	lc *_Cache
}

type _K struct {
	Key    string
	AK     string
	SK     string
	UID    uint32
	Bucket string
}

func (k _K) String() string { return k.Key }

type _V struct {
	Zone   string
	IOHost string
}

func (h *qiniuAdminHandlerV2) Get(ctx context.Context, args Request, opts ...GetOption,
) (resp *Response, err error) {

	for _, opt := range opts {
		opt(&args)
	}

	uri := args.URI
	var bucket, key string
	var uid uint32
	{
		// u: Scheme: "qiniu", Host: "z0", Path: "/test/1.png",
		u, err := url.Parse(uri)
		if err != nil {
			return nil, errors.Wrap(err, "url.Parse")
		}
		subStr := pathRegex.FindStringSubmatch(u.Path)
		if len(subStr) != 3 {
			return nil, ErrBadUri
		}
		bucket = subStr[1]
		key = subStr[2]
		// uid = u.User.Username()
		uid64, err := strconv.ParseUint(u.User.Username(), 10, 32)
		if err != nil {
			return nil, err
		}
		uid = uint32(uid64)
	}

	var xl = xlog.FromContextSafe(ctx)

	ak, sk, err := auth.AkSk(h.qCli, uid)
	if err != nil {
		xl.Warnf("get aksk failed. %d %v", uid, err)
		return nil, err
	}

	var zone, ioHost string
	if h.lc != nil {
		v0, err := h.lc.Get(context.Background(),
			_K{Key: fmt.Sprintf("%d-%s", uid, bucket), UID: uid, Bucket: bucket, AK: ak, SK: sk})
		if err != nil {
			return nil, err
		}
		v := v0.(_V)
		zone = v.Zone
		ioHost = v.IOHost
	} else {
		info := struct {
			Zone string `json:"zone"`
		}{}
		err = rpc.NewClientWithTransport(qboxmac.NewTransport(&qboxmac.Mac{AccessKey: ak, SecretKey: []byte(sk)}, nil)).
			Call(ctx, &info, "GET", fmt.Sprintf("http://%s/bucket/%s", h.RSHost, bucket))
		if err != nil {
			return nil, err
		}

		zone = info.Zone
		ioHost = h.IOHosts[zone]
	}

	if ioHost == "" {
		return nil, errors.New("not support zone")
	}
	host := fmt.Sprintf("%s-%s.%s.src.qbox.me",
		hex.EncodeToString([]byte(bucket)), strconv.FormatUint(uint64(uid), 36), zone)

	var uri2 = kodo.New(0, &kodo.Config{AccessKey: ak, SecretKey: sk}).MakePrivateUrl(kodo.MakeBaseUrl(host, key), nil)
	{
		uri3, _ := url.Parse(uri2)
		uri3.Host = ioHost
		uri2 = uri3.String()
	}
	req, err := http.NewRequest("GET", uri2, nil)
	if err != nil {
		return nil, errors.Wrap(err, "getHTTP http.NewRequest")
	}
	req.Host = host

	if args.beginOff != nil {
		FormatRangeRequest(req.Header, args)
	}

	req = req.WithContext(ctx)
	client := h.Client
	if client == nil {
		client = http.DefaultClient
	}
	_resp, err := client.Do(req)
	if err != nil {
		return nil, errors.Wrap(err, "getHTTP client.Do")
	}
	if _resp.StatusCode/100 != 2 {
		defer _resp.Body.Close()
		return nil, rpc.ResponseError(_resp)
	}
	return transResp(_resp), nil
}

func (h *qiniuAdminHandlerV2) Names() []string {
	return []string{"qiniu"}
}

////////////////////////////////////////////////////////////////////////////////

type StaticHeader struct {
	http.Header
	RT http.RoundTripper
}

func (t StaticHeader) RoundTrip(req *http.Request) (resp *http.Response, err error) {
	for key := range t.Header {
		req.Header.Set(key, t.Header.Get(key))
	}
	return t.RT.RoundTrip(req)
}

////////////////////////////////////////////////////////////////////////////////

type CacheConfig struct {
	Expires     int `json:"expires_ms"`  // 如果缓存超过这个时间，则强制刷新(防止取到太旧的值)，以毫秒为单位。
	Duration    int `json:"duration_ms"` // 如果缓存没有超过这个时间，不要去刷新(防止刷新过于频繁)，以毫秒为单位。
	PoolSize    int `json:"pool_size"`
	ChanBufSize int `json:"chan_bufsize"` // 异步消息队列缓冲区大小。
}

type _Key interface {
	String() string
}

type _Item struct {
	Key   _Key
	Value interface{}
	T     time.Time
}

type _Cache struct {
	CacheConfig

	Expires  time.Duration
	Duration time.Duration
	Fetch    func(context.Context, _Key) (interface{}, error)

	m  map[string]*list.Element // _Item
	l  *list.List               // _Item
	ch chan *list.Element
	*sync.Mutex
}

func newCache(conf CacheConfig, fetch func(context.Context, _Key) (interface{}, error)) *_Cache {
	if conf.Expires == 0 {
		conf.Expires = 60 * 60 * 1000
	}
	if conf.Duration == 0 {
		conf.Duration = 5 * 60 * 1000
	}
	if conf.PoolSize == 0 {
		conf.PoolSize = 1024
	}
	if conf.ChanBufSize == 0 {
		conf.ChanBufSize = 64
	}
	c := &_Cache{
		CacheConfig: conf,
		Expires:     time.Millisecond * time.Duration(conf.Expires),
		Duration:    time.Millisecond * time.Duration(conf.Duration),
		Fetch:       fetch,
		m:           make(map[string]*list.Element),
		l:           list.New(),
		ch:          make(chan *list.Element, conf.ChanBufSize),
		Mutex:       new(sync.Mutex),
	}
	go c.run()
	return c
}

func (c *_Cache) run() {
	for item := range c.ch {
		now := time.Now()
		c.Lock()
		v := item.Value.(_Item)
		k := v.Key.String()
		d := now.Sub(v.T)
		if _, ok := c.m[k]; !ok {
			c.Unlock()
			continue
		}
		c.Unlock()
		if d < c.Duration {
			continue
		}

		v2, err := c.Fetch(context.Background(), v.Key)
		if err != nil {
			c.Lock()
			delete(c.m, k)
			c.l.Remove(item)
			c.Unlock()
			continue
		}

		now = time.Now()
		c.Lock()
		v.Value = v2
		v.T = now
		item.Value = v
		c.Unlock()

	}
}

func (c *_Cache) Get(ctx context.Context, key _Key) (interface{}, error) {
	xl := xlog.FromContextSafe(ctx)
	_ = xl

	now := time.Now()
	k := key.String()
	c.Lock()
	item, ok := c.m[k]
	if ok {
		v := item.Value.(_Item)
		d := now.Sub(v.T)
		if d < c.Expires {
			c.Unlock()
			if d >= c.Duration {
				c.ch <- item
			}
			return v.Value, nil
		}
		delete(c.m, k)
		c.l.Remove(item)
	}
	c.Unlock()

	v, err := c.Fetch(ctx, key)
	if err != nil {
		return v, err
	}

	c.Lock()
	item = c.l.PushBack(_Item{Key: key, Value: v, T: now})
	c.m[k] = item
	if c.l.Len() > c.PoolSize {
		c.l.Remove(c.l.Front())
	}
	c.Unlock()

	return v, nil
}
