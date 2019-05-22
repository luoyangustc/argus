package lb

import (
	"math/rand"
	"net"
	"net/http"
	"net/url"
	"sync"
	"sync/atomic"
	"time"

	"github.com/golang/groupcache/singleflight"
	"github.com/qiniu/log.v1"
	"github.com/qiniu/xlog.v1"

	"qbox.us/ratelimit"
)

// --------------------------------------------------------------------

func init() {
	rand.Seed(time.Now().UnixNano())
}

func randomShrink(ss []*host) ([]*host, *host) {
	n := len(ss)
	if n == 1 {
		return ss[0:0], ss[0]
	}
	i := rand.Intn(n)
	s := ss[i]
	ss[i] = ss[0]
	return ss[1:], s
}

// --------------------------------------------------------------------

type host struct {
	raw            string
	URL            *url.URL
	rl             *ratelimit.RateLimiter
	host           string
	lastFailedTime int64

	punishLock  sync.RWMutex
	punishReqId string
}

func (h *host) SetFail(xl *xlog.Logger) {
	if h.rl.Limit() {
		failTime := time.Now()
		atomic.StoreInt64(&h.lastFailedTime, failTime.Unix())
		h.punishLock.Lock()
		h.punishReqId = xl.ReqId()
		h.punishLock.Unlock()
		xl.Printf("host.SetFail, host: %v", h.raw)
	}
}

func (h *host) IsPunished(failRetryInterval int64) (punishReqId string, ok bool) {
	lastFailedTime := atomic.LoadInt64(&h.lastFailedTime)
	isPunished := lastFailedTime != 0 && time.Now().Unix()-lastFailedTime < failRetryInterval
	if isPunished {
		h.punishLock.RLock()
		punishReqId = h.punishReqId
		h.punishLock.RUnlock()
	}
	return punishReqId, isPunished
}

// --------------------------------------------------------------------

type retrySelector struct {
	idx               uint32  // 被排除的 host idx
	hosts             []*host // 完整 host 列表
	retryHosts        []*host
	failRetryInterval int64
}

func (s *retrySelector) Get(xl *xlog.Logger) (h *host) {
	if len(s.retryHosts) == 0 {
		s.retryHosts = make([]*host, 0, len(s.hosts)-1)
		for i, h := range s.hosts {
			if i == int(s.idx) {
				continue
			}
			if punishReqId, ok := h.IsPunished(s.failRetryInterval); ok {
				xl.Printf("retrySelector.Get(),  %v is during punishtime, punish reqid : %v", h.raw, punishReqId)
				continue
			}
			s.retryHosts = append(s.retryHosts, h)
		}
	}
	if len(s.retryHosts) == 0 {
		return
	}
	// 不再管是否在失败列表中
	s.retryHosts, h = randomShrink(s.retryHosts)
	return
}

// --------------------------------------------------------------------

type selector struct {
	tryTimes          uint32
	failRetryInterval int64 // 被屏蔽的时间，-1 时忽略屏蔽
	dnsResolve        bool
	dnsCacheTimeS     int64
	lookupHost        func(host string) ([]string, error)
	oriHosts          []*host

	hostsMu             sync.RWMutex // protect hosts
	hosts               []*host
	hostsLastUpdateTime int64
	g                   singleflight.Group

	reqHostMu sync.RWMutex            // protect reqHost
	reqHost   map[*http.Request]*host // 在有代理的时候使用

	idx uint32
}

func newSelector(hosts []string, tryTimes uint32, failRetryInterval int64, dnsResolve bool, dnsCacheTimeS int64, lookupHost func(host string) ([]string, error), maxFails int, maxFailsPeriods int64) *selector {
	if len(hosts) == 0 {
		log.Panic("empty hosts")
	}
	if failRetryInterval == 0 {
		failRetryInterval = DefaultFailRetryInterval
	}
	if dnsCacheTimeS == 0 {
		dnsCacheTimeS = DefaultDnsCacheTimeS
	}
	if lookupHost == nil {
		lookupHost = net.LookupHost
	}
	var hs []*host
	for _, h := range hosts {
		u, err := url.Parse(h)
		if err != nil {
			log.Panic("error host", h, err)
		}
		rl := ratelimit.New(maxFails-1, time.Duration(maxFailsPeriods)*time.Second)
		hs = append(hs, &host{URL: u, raw: h, rl: rl})
	}
	s := &selector{
		tryTimes:          tryTimes,
		failRetryInterval: failRetryInterval,
		dnsResolve:        dnsResolve,
		dnsCacheTimeS:     dnsCacheTimeS,
		lookupHost:        lookupHost,
		oriHosts:          hs,
		hosts:             hs,
		reqHost:           make(map[*http.Request]*host),
	}
	if s.dnsResolve {
		err := s.resolveDns()
		if err != nil {
			panic("s.resloveDns failed: " + err.Error())
		}
		s.hostsLastUpdateTime = time.Now().UnixNano()
	}
	return s
}

func (s *selector) GetTryTimes() uint32 {
	if s.tryTimes != 0 {
		return s.tryTimes
	}
	s.hostsMu.RLock()
	t := len(s.hosts)
	s.hostsMu.RUnlock()
	return uint32(t)
}

func (s *selector) resolveOne(h *host) ([]*host, error) {
	if h.URL.Scheme == "https" {
		return []*host{
			{raw: h.raw, URL: h.URL, rl: h.rl},
		}, nil
	}
	domain, port, err := net.SplitHostPort(h.URL.Host)
	if err != nil {
		domain, port = h.URL.Host, ""
	}

	oriAddrs, err := s.lookupHost(domain)
	log.Printf("resolveDns, domain: %v, addrs: %v, err: %v\n", domain, oriAddrs, err)
	if err != nil {
		return nil, err
	}
	addrs := make([]string, len(oriAddrs))
	for i := range oriAddrs {
		addrs[i] = h.URL.Scheme + "://" + oriAddrs[i]
		if port != "" {
			addrs[i] += ":" + port
		}
	}
	hs := make([]*host, len(addrs))
	for i, raw := range addrs {
		u, err := url.Parse(raw)
		if err != nil {
			return nil, err
		}
		hs[i] = &host{URL: u, raw: raw, rl: h.rl, host: h.URL.Host}
	}
	return hs, nil
}

func (s *selector) resolveDns() error {
	var hs []*host
	for _, h := range s.oriHosts {
		hs2, err := s.resolveOne(h)
		if err != nil {
			return err
		}
		hs = append(hs, hs2...)
	}
	// carry on host info,
	// resolve之后需要保留相同IP上次失败的时间，和失败的频率信息
	m := make(map[string]*host)
	for _, h := range s.hosts {
		m[h.raw] = h
	}
	for i, h := range hs {
		if t, ok := m[h.raw]; ok {
			hs[i] = t
		}
	}

	s.hostsMu.Lock()
	s.hosts = hs
	s.hostsMu.Unlock()
	return nil
}

func (s *selector) Get(xl *xlog.Logger) (h *host, rs *retrySelector) {

	now := time.Now().UnixNano()
	lastUpdate := atomic.LoadInt64(&s.hostsLastUpdateTime)
	if s.dnsResolve && now-lastUpdate > s.dnsCacheTimeS*1e9 {
		go s.g.Do("", func() (interface{}, error) {
			lastUpdate := atomic.LoadInt64(&s.hostsLastUpdateTime)
			if now-lastUpdate > s.dnsCacheTimeS*1e9 {
				err := s.resolveDns()
				if err == nil {
					atomic.StoreInt64(&s.hostsLastUpdateTime, time.Now().UnixNano())
				} else {
					xl.Error("s.resloveDns failed", err)
				}
			}
			return nil, nil
		})
	}
	s.hostsMu.RLock()
	hs := s.hosts
	s.hostsMu.RUnlock()

	var idx uint32
	for i := 0; i < len(hs); i++ {
		idx = atomic.AddUint32(&s.idx, 1) % uint32(len(hs))
		if punishReqId, ok := hs[idx].IsPunished(s.failRetryInterval); !ok {
			h = hs[idx]
			break
		} else {
			xl.Printf("selector.Get(), %v is during punishtime, punish reqid :%v", hs[idx].raw, punishReqId)
		}
	}
	if h == nil {
		return
	}
	rs = &retrySelector{idx: idx, hosts: hs, failRetryInterval: s.failRetryInterval}
	return
}

func (s *selector) SetReqHost(req *http.Request, h *host) {
	s.reqHostMu.Lock()
	defer s.reqHostMu.Unlock()

	s.reqHost[req] = h
}

func (s *selector) GetReqHost(req *http.Request) (h *host, ok bool) {
	s.reqHostMu.RLock()
	defer s.reqHostMu.RUnlock()

	h, ok = s.reqHost[req]
	return
}

func (s *selector) DelReqHost(req *http.Request) {
	s.reqHostMu.Lock()
	defer s.reqHostMu.Unlock()

	delete(s.reqHost, req)
}

// --------------------------------------------------------------------
