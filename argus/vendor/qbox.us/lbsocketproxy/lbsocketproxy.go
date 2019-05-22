package lbsocketproxy

import (
	"math/rand"
	"net"
	"sync/atomic"
	"time"

	"github.com/qiniu/log.v1"
	"qbox.us/iputil"

	"golang.org/x/net/proxy"
)

type LbSocketProxy struct {
	conf           Config
	proxies        []proxy.Dialer
	idx            uint32
	ShouldUseProxy ShouldUseProxy
}

/*
type:
	all: 所有请求走代理
	default: 出idc请求走代理
*/

type Config struct {
	Hosts         []string    `json:"hosts"`
	DialTimeoutMs int         `json:"dial_timeout_ms"`
	TryTimes      int         `json:"try_times"`
	Auth          *proxy.Auth `json:"auth"`
	Type          string      `json:"type"`
}

type ShouldUseProxy func(dstIP string) bool

var AllUseProxy = func(dstIP string) bool { return true }

func NewLbSocketProxy(conf *Config) (lbs *LbSocketProxy, err error) {
	if conf.TryTimes == 0 {
		conf.TryTimes = len(conf.Hosts)
	}
	var proxies []proxy.Dialer
	for i := 0; i < len(conf.Hosts); i++ {
		forward := &net.Dialer{Timeout: time.Millisecond * time.Duration(conf.DialTimeoutMs)}
		p, err := proxy.SOCKS5("tcp", conf.Hosts[i], conf.Auth, forward)
		if err != nil {
			return nil, err
		}
		proxies = append(proxies, p)
	}
	lbs = &LbSocketProxy{
		conf:    *conf,
		proxies: proxies,
	}
	if conf.Type == "all" {
		lbs.ShouldUseProxy = AllUseProxy
	} else {
		lbs.ShouldUseProxy = iputil.NewDefaultIdcIpChecker().IsNotInSameIDC
	}
	return
}

func init() {
	rand.Seed(time.Now().UnixNano())
}

func randomIdx(failedCount int, all []bool) int {
	if failedCount >= len(all) {
		return rand.Intn(len(all))
	}
	idx := rand.Intn(len(all) - failedCount)
	for ok, i := 0, 0; i < len(all); i++ {
		if !all[i] {
			if ok == idx {
				return i
			}
			ok += 1
		}
	}
	return rand.Intn(len(all))
}

func (self *LbSocketProxy) Dial(addr net.Addr) (c net.Conn, err error) {
	if !self.ShouldUseProxy(addr.(*net.TCPAddr).IP.String()) {
		timeout := time.Millisecond * time.Duration(self.conf.DialTimeoutMs)
		return (&net.Dialer{Timeout: timeout}).Dial("tcp", addr.String())
	}
	idx := int(atomic.AddUint32(&self.idx, 1)) % len(self.proxies)
	var all []bool
	for i := 1; i <= self.conf.TryTimes; i++ {
		c, err = self.proxies[idx].Dial("tcp", addr.String())
		if err == nil {
			log.Debugf("connect to %s use proxy %v success with local addr %v", addr.String(), c.RemoteAddr(), c.LocalAddr())
			return
		}
		if _, ok := err.(net.Error); !ok {
			break
		}
		log.Warnf("connect to %v with proxy %v failed: %v", addr.String(), self.conf.Hosts[idx], err)
		if len(all) == 0 {
			all = make([]bool, len(self.conf.Hosts))
		}
		all[idx] = true // mark failed
		idx = randomIdx(i, all)
	}
	return
}
