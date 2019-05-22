package uuid

import (
	"crypto/rand"
	"encoding/base64"
	"io"
	"sync/atomic"
	"syscall"
	"time"
)

const minGuidLen = 12

// ---------------------------------------------------------------------------

var g_incVal uint32

func init() {

	g_incVal = uint32(time.Now().UnixNano() / 1e6)
}

func Make(n int) (v []byte, err error) {

	if n < minGuidLen {
		return nil, syscall.EINVAL
	}

	incVal := atomic.AddUint32(&g_incVal, 1)

	v = make([]byte, n)
	v[0] = byte(incVal)
	v[1] = byte(incVal >> 8)
	_, err = io.ReadFull(rand.Reader, v[2:])
	return
}

func Gen(n int) (s string, err error) {

	v, err := Make(n)
	if err != nil {
		return
	}

	s = base64.URLEncoding.EncodeToString(v)
	return
}

// ---------------------------------------------------------------------------
