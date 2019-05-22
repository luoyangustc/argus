package uri

import (
	"net"
	"net/http"
	"strings"
)

/*
IsPublicIP 判断一个IP是否是公网IP
里面的七牛内网网段 reviewed by gaolei@qiniu.com
*/
func IsPublicIP(ip string) bool {
	innets := []string{"10.0.0.0/8", "100.64.0.0/10", "172.16.0.0/12", "192.168.0.0/16", "127.0.0.0/8"}
	for _, innet := range innets {
		_, subnet, _ := net.ParseCIDR(innet)
		ipv := net.ParseIP(ip)
		if ipv == nil {
			return false
		}
		if subnet.Contains(ipv) {
			return false
		}
		// ipv6
		if strings.Contains(ipv.String(), ":") {
			return false
		}
	}
	return true
}

// OnlyPublicIPHTTPClient 返回一个只允许访问公网资源的http client
func OnlyPublicIPHTTPClient() http.RoundTripper {
	dialFunc := func(network, address string) (conn net.Conn, err error) {
		conn, err = net.Dial(network, address)
		if err != nil {
			return
		}
		remoteAddr := conn.RemoteAddr().String()
		if h1, _, err := net.SplitHostPort(remoteAddr); err == nil {
			remoteAddr = h1
		}
		if !IsPublicIP(remoteAddr) {
			conn.Close()
			return
		}
		return
	}
	return &http.Transport{
		Dial: dialFunc,
	}
}
