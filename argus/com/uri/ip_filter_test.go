package uri

import (
	"github.com/stretchr/testify.v2/assert"
	"testing"
)

func TestIsPublicIP(t *testing.T) {
	a := []string{
		"192.168.201.243", "192.168.200.26", "10.200.20.54", "10.34.37.56", //内网地址
		"", "xx", // 无效地址
		"127.0.0.2", "127.0.0.1", //local
		"[::1]", "::1", //ipv6
		"180.149.132.47", // true
	}
	b := []bool{
		false, false, false, false,
		false, false,
		false, false,
		false, false,
		true,
	}
	for i, v := range a {
		assert.Equal(t, IsPublicIP(v), b[i], "%v %v", i, v)
	}
}
