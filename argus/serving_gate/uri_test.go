package gate

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestImproveURI(t *testing.T) {
	{
		uri2, _ := improveURI("http://qiniu.com", 1)
		assert.Equal(t, "http://qiniu.com", uri2)
	}
	{
		uri2, _ := improveURI("qiniu://z0/bucket/key", 1)
		assert.Equal(t, "qiniu://1@z0/bucket/key", uri2)
	}
}

func TestBadURI(t *testing.T) {
	assert.False(t, isBadURI("file:///xxx"))
	assert.False(t, isBadURI("http://xx/xx"))
	assert.False(t, isBadURI("https://xx/xx"))
	assert.False(t, isBadURI("sts://xx/xx"))
	assert.False(t, isBadURI("qiniu://xx/xx"))
	assert.False(t, isBadURI("data:application/octet-stream;base64,xxx"))
	assert.True(t, isBadURI("data:application/octet-stream:base64,xxx"))
}

func TestDataURI(t *testing.T) {
	assert.True(t, isDataURI("data:application/octet-stream;base64,xxx"))
	bs, err := decodeDataURI(encodeDataURI([]byte("abc")))
	assert.NoError(t, err)
	assert.Equal(t, "abc", string(bs))
}
