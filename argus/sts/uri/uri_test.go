package uri

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestURI(t *testing.T) {
	assert.Equal(t, NONE, TypeOf(NewURI("data:application/octet-stream:base64,MTIz")))
}

func TestSchemeOfURI(t *testing.T) {
	assert.Equal(t, "file", SchemeOf("file://xxxx"))
	assert.Equal(t, "http", SchemeOf("http://qiniu.com"))
	assert.Equal(t, "http", SchemeOf("https://qiniu.com"))
	assert.Equal(t, "sts", SchemeOf("sts://ip:port/xx"))
	assert.Equal(t, "qiniu", SchemeOf("qiniu://z0/bucket/key"))
	assert.Equal(t, "data", SchemeOf("data:application/octet-stream;base64,xxx"))
}
