package fetcher

import (
	"encoding/base64"
	"encoding/binary"
	"encoding/hex"
	"net"
)

func ParseIpPortFromBase64(str string) (net.IP, uint16) {
	ip, _ := base64.StdEncoding.DecodeString(str)
	lenIp := len(ip)
	if lenIp == 6 || lenIp == 18 {
		return net.IP(ip[:lenIp-2]), binary.LittleEndian.Uint16(ip[lenIp-2:])
	} else {
		return net.IP(ip), 0
	}
}

func ParseMD5FromBase64(str string) string {
	buf, err := base64.StdEncoding.DecodeString(str)
	if err != nil || len(buf) == 0 {
		return ""
	}

	return hex.EncodeToString(buf)
}
