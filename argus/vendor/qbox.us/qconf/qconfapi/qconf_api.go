// 整个客户端只能处理服务端返回为bson的情况(changed at 2016/11/28)
package qconfapi

import (
	"io/ioutil"
	"sync/atomic"
	"time"

	"github.com/bradfitz/gomemcache/memcache"
	"github.com/qiniu/api/auth/digest"
	"github.com/qiniu/rpc.v1"
	brpc "github.com/qiniu/rpc.v1/brpc/lb.v2.1"
	"github.com/qiniu/rpc.v1/lb.v2.1"
	"github.com/qiniu/xlog.v1"
	"labix.org/v2/mgo/bson"
)

// ------------------------------------------------------------------------

const (
	Cache_Normal      = 0
	Cache_NoSuchEntry = 1
)

type LBConfig struct {
	brpc.Config
	Transport lb.TransportConfig `json:"transport"`
}

type MasterConfig struct {
	Default  LBConfig `json:"default"`
	Failover LBConfig `json:"failover"`
}

type Config struct {
	McHosts []string `json:"mc_hosts"` // 如果是管理api(比如put/rm/refresh)，则不需要 McHosts

	// 兼容老的配置
	MasterHosts []string `json:"master_hosts"`

	// 推荐使用新的配置
	Master MasterConfig `json:"master"`

	AccessKey string `json:"access_key"` // 如果不需要授权，可以没有这个值
	SecretKey string `json:"secret_key"`

	LcacheExpires     int `json:"lc_expires_ms"`   // 如果缓存超过这个时间，则强制刷新(防止取到太旧的值)，以毫秒为单位。
	LcacheDuration    int `json:"lc_duration_ms"`  // 如果缓存没有超过这个时间，不要去刷新(防止刷新过于频繁)，以毫秒为单位。
	LcacheChanBufSize int `json:"lc_chan_bufsize"` // 异步消息队列缓冲区大小。

	McExpires   int32 `json:"mc_expires_s"`     // memcache item的失效时间，最多一个月，以秒为单位，0为不失效
	McRWTimeout int64 `json:"mc_rw_timeout_ms"` // memcache 客户端的读写超时时间，以毫秒为单位。
}

type Client struct {
	Lcache     *localCache
	mcaches    []*memcache.Client
	mcIdxBegin uint32
	McExpires  int32
	Conn       *brpc.Client
}

func New(cfg *Config) *Client {

	mcaches := make([]*memcache.Client, len(cfg.McHosts))
	for i, host := range cfg.McHosts {
		client := memcache.New(host)
		client.Timeout = time.Duration(cfg.McRWTimeout) * time.Millisecond
		mcaches[i] = client
	}
	oneMonth := int32(30 * 24 * 60 * 60)
	if cfg.McExpires > oneMonth {
		panic("cfg.McExpires should be up to 1 month")
	}

	p := &Client{
		mcaches:   mcaches,
		McExpires: cfg.McExpires,
	}

	master := &cfg.Master

	if len(master.Default.Hosts) == 0 {
		master.Default.Hosts = cfg.MasterHosts
	}
	setMasterDefaultConfig(&master.Default)
	setMasterDefaultConfig(&master.Failover)

	defaultTr := lb.NewTransport(&master.Default.Transport)
	failoverTr := lb.NewTransport(&master.Failover.Transport)

	if cfg.AccessKey != "" {
		mac := &digest.Mac{
			AccessKey: cfg.AccessKey,
			SecretKey: []byte(cfg.SecretKey),
		}
		defaultTr = digest.NewTransport(mac, defaultTr)
		failoverTr = digest.NewTransport(mac, failoverTr)
	}

	if len(master.Failover.Hosts) > 0 {
		p.Conn = brpc.NewWithFailover(&master.Default.Config, &master.Failover.Config, defaultTr, failoverTr, nil)
	} else {
		p.Conn = brpc.New(&master.Default.Config, defaultTr)
	}

	if cfg.LcacheExpires > 0 {
		p.Lcache = newLocalCache(
			int64(cfg.LcacheExpires)*1e6, int64(cfg.LcacheDuration)*1e6,
			cfg.LcacheChanBufSize, p.getBytesNolc)
	}
	return p
}

func setMasterDefaultConfig(cfg *LBConfig) {
	if cfg.TryTimes == 0 {
		cfg.TryTimes = uint32(len(cfg.Hosts))
	}
	if cfg.FailRetryIntervalS == 0 {
		cfg.FailRetryIntervalS = -1
	}

	if cfg.Transport.DialTimeoutMS == 0 {
		cfg.Transport.DialTimeoutMS = 1000
	}
	if cfg.Transport.TryTimes == 0 {
		cfg.Transport.TryTimes = uint32(len(cfg.Transport.Proxys))
	}
	if cfg.Transport.FailRetryIntervalS == 0 {
		cfg.Transport.FailRetryIntervalS = -1
	}
}

// ------------------------------------------------------------------------

type insArgs struct {
	Doc interface{} `bson:"doc"`
}

func (p *Client) Insert(l rpc.Logger, doc interface{}) (err error) {
	return p.Conn.CallWithBson(l, nil, "/insb", insArgs{doc})
}

// ------------------------------------------------------------------------

type putArgs struct {
	Id     string      `bson:"id"`
	Change interface{} `bson:"chg"`
}

type M map[string]interface{}

func (p *Client) Modify(l rpc.Logger, id string, change interface{}) (err error) {
	return p.Conn.CallWithBson(l, nil, "/putb", putArgs{id, change})
}

func (p *Client) SetProp(l rpc.Logger, id, prop string, val interface{}) (err error) {

	return p.Modify(l, id, M{"$set": M{prop: val}})
}

func (p *Client) DeleteProp(l rpc.Logger, id, prop string) (err error) {

	return p.Modify(l, id, M{"$unset": M{prop: 1}})
}

// ------------------------------------------------------------------------

func (p *Client) Delete(l rpc.Logger, id string) (err error) {
	return p.Conn.CallWithForm(l, nil, "/rm", map[string][]string{
		"id": {id},
	})
}

// ------------------------------------------------------------------------

func (p *Client) Refresh(l rpc.Logger, id string) (err error) {
	return p.Conn.CallWithForm(l, nil, "/refresh", map[string][]string{
		"id": {id},
	})
}

// ------------------------------------------------------------------------

func (p *Client) GetFromLc(l rpc.Logger, ret interface{}, id string, cacheFlags int) (exist bool, err error) {

	log := xlog.NewWith(l)
	exist, err = p.getFromLc(log, ret, id, cacheFlags)
	return
}

func (p *Client) GetFromMaster(l rpc.Logger, ret interface{}, id string, cacheFlags int) (err error) {

	log := xlog.NewWith(l)
	p.deleteFromMcache(log, id)
	if p.Lcache != nil {
		p.Lcache.deleteItemSafe(id)
	}
	return p.Get(log, ret, id, cacheFlags)
}

func (p *Client) Get(l rpc.Logger, ret interface{}, id string, cacheFlags int) (err error) {

	log := xlog.NewWith(l)
	err = p.get(log, ret, id, cacheFlags)
	return
}

func (p *Client) getFromLc(log *xlog.Logger, ret interface{}, id string, cacheFlags int) (exist bool, err error) {

	var exp bool
	if p.Lcache != nil {
		exist, exp, err = p.Lcache.getFromLc(log, ret, id, cacheFlags, time.Now().UnixNano())
		if err != nil {
			return
		}
		if exp {
			p.Lcache.updateItem(log, id, cacheFlags)
		}
	}
	return
}

func (p *Client) get(log *xlog.Logger, ret interface{}, id string, cacheFlags int) (err error) {

	if p.Lcache != nil {
		err = p.Lcache.get(log, ret, id, cacheFlags, time.Now().UnixNano())
		return
	}

	val, err := p.getBytesNolc(log, id, cacheFlags)
	if err != nil {
		return
	}
	err = bson.Unmarshal(val, ret)
	if err != nil {
		log.Error("qconf.Get: bson.Unmarshal failed -", err)
	}
	return
}

func (p *Client) getBytesNolc(log *xlog.Logger, id string, cacheFlags int) (val []byte, err error) {

	code, val, err := p.getFromMcache(log, id)
	if err == nil {
		goto done
	}
	code, val, err = p.getFromMaster(log, "/getb", id, cacheFlags)
	if err == nil {
		goto done
	}
	return

done:
	if code != 200 {
		err = &rpc.ErrorInfo{
			Code: code,
			Err:  string(val),
		}
	}
	return
}

func (p *Client) deleteFromMcache(log *xlog.Logger, id string) (err error) {

	for i, mc := range p.mcaches {
		err2 := mc.Delete(id)
		if err2 != nil && err2 != memcache.ErrCacheMiss {
			err = err2
			log.Warn("qconf.getFromMcache failed:", i, id, err2)
		}
	}
	return
}

func (p *Client) getFromMcache(log *xlog.Logger, id string) (int, []byte, error) {

	N := uint32(len(p.mcaches))
	idxBegin := atomic.LoadUint32(&p.mcIdxBegin)
	for idx := uint32(0); idx < N; idx++ {
		i := (idxBegin + idx) % N
		mc := p.mcaches[i]
		item, err := mc.Get(id)
		if err == nil {
			return int(item.Flags), item.Value, nil
		}
		if err == memcache.ErrCacheMiss {
			// 在遇到miss时交换使用的cache，可以减少mc重启带来的冲击
			atomic.AddUint32(&p.mcIdxBegin, 1)
			return 0, nil, err
		}
		log.Warn("qconf.getFromMcache failed:", i, id, err)
	}
	return 0, nil, memcache.ErrNoServers
}

func (p *Client) getFromMaster(log *xlog.Logger, url string, id string, cacheFlags int) (code int, b []byte, err error) {

	resp, err := p.Conn.PostWithForm(log, url, map[string][]string{
		"id": {id},
	})
	if err != nil {
		log.Warn("qconf.getFromMaster: post form failed -", url, id, err)
		return
	}
	defer resp.Body.Close()

	code = resp.StatusCode
	if code != 200 {
		b = []byte(rpc.ResponseError(resp).Error())
		switch cacheFlags {
		case Cache_NoSuchEntry:
			if code != 612 && code != 404 {
				return
			}
		default:
			return
		}
	} else {
		b, err = ioutil.ReadAll(resp.Body)
		if err != nil {
			log.Warn("qconf.getFromMaster: read resp.Body failed -", url, id, err)
			return
		}
	}

	item := &memcache.Item{
		Key:        id,
		Value:      b,
		Flags:      uint32(code),
		Expiration: p.McExpires,
	}
	for i, mc := range p.mcaches {
		err2 := mc.Set(item)
		if err2 != nil {
			err3 := mc.Delete(id) // 对mc脏数据的最后一次努力
			log.Warn("qconf.getFromMaster: put memcache failed -", i, err2, "delete:", err3)
		}
	}
	return
}

// ------------------------------------------------------------------------
