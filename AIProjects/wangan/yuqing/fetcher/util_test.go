package fetcher

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestParseMD5FromBase64(t *testing.T) {
	b64 := "XzbATbAVBOjNBSXiOWAxiQ=="
	md5 := ParseMD5FromBase64(b64)
	assert.Equal(t, "5f36c04db01504e8cd0525e239603189", md5)
}
