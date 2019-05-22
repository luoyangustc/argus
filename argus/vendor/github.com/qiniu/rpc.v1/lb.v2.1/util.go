package lb

import (
	"io"
	"io/ioutil"
)

// https://pm.qbox.me/issues/24188#note-10
// go >= 1.5 roundtrip 同一个 request （重试）会导致后续的请求返回 error request canceled.
// 需要读完 resp.Body 才不会出现错误，而且不读完 resp.Body 无法重用链接。
func discardAndClose(r io.ReadCloser) error {
	io.Copy(ioutil.Discard, r)
	return r.Close()
}
