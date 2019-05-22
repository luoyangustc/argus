package iputil

import (
	"net"
	"sync"
)

var idcIPMap = map[string]string{
	"bc":        "10.30.0.0/15",   // 昌平
	"fs":        "10.44.0.0/15",   // 佛山
	"gz":        "10.42.0.0/15",   // 广州
	"lac":       "10.40.0.0/15",   // 北美
	"nb":        "192.168.0.0/17", // 宁波
	"ns":        "10.36.0.0/15",   // 南沙
	"tc":        "10.32.0.0/15",   // 太仓
	"xs":        "10.34.0.0/15",   // 下沙
	"local":     "127.0.0.1/8",    // 本地
	"cs_dev":    "10.200.20.0/24", // 测试dev环境
	"cs_dev_xs": "10.200.30.0/24", // 测试dev_xs环境
	// "cs":    "10.200.0.0/15",  // 测试
}

type IdcIpChecker struct {
	idcNets map[string]*net.IPNet
	myIDC   string
	lock    sync.RWMutex
}

func NewDefaultIdcIpChecker() (checker *IdcIpChecker) {
	checker, err := NewIpCheckerWithIpMap(idcIPMap)
	if err != nil {
		panic(err)
	}
	return checker
}

func NewIpCheckerWithIpMap(idcIPMap map[string]string) (checker *IdcIpChecker, err error) {
	idcNets := make(map[string]*net.IPNet)
	for idc, ipRange := range idcIPMap {
		_, ipNet, err := net.ParseCIDR(ipRange)
		if err != nil {
			return nil, err
		}
		idcNets[idc] = ipNet
	}
	checker = &IdcIpChecker{
		idcNets: idcNets,
	}
	return
}

func (c *IdcIpChecker) GetIDCByIP(ip string) (idc string) {
	ipParsed := net.ParseIP(ip)
	for idc, ipNet := range c.idcNets {
		if ipNet.Contains(ipParsed) {
			return idc
		}
	}
	return ""
}

func (c *IdcIpChecker) SetMyIDC(idc string) {
	c.lock.Lock()
	c.myIDC = idc
	c.lock.Unlock()
}

func (c *IdcIpChecker) GetMyIDC() (idc string) {
	c.lock.RLock()
	if c.myIDC != "" {
		c.lock.RUnlock()
		return c.myIDC
	}
	c.lock.RUnlock()
	ips, err := net.InterfaceAddrs()
	if err != nil {
		return
	}
	for _, ip := range ips {
		ip, _, err := net.ParseCIDR(ip.String())
		if err != nil {
			return
		}
		idcTmp := c.GetIDCByIP(ip.String())
		if idcTmp != "local" && idcTmp != "" {
			c.myIDC = idcTmp
			return idcTmp
		}
	}
	c.lock.Lock()
	c.myIDC = "local"
	c.lock.Unlock()
	return c.myIDC
}

func (c *IdcIpChecker) IsInSameIDC(dstIP string) bool {
	return c.GetMyIDC() == c.GetIDCByIP(dstIP)
}

func (c *IdcIpChecker) IsNotInSameIDC(dstIP string) bool {
	return !c.IsInSameIDC(dstIP)
}
