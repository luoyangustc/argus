package lb

import (
	"math/rand"
	"net/http"
	"time"
)

func init() {
	rand.Seed(time.Now().UnixNano())
}

func copyExcept(ss []string, i int) []string {

	n := len(ss)
	if n == 1 {
		return []string{}
	}
	ret := make([]string, 0, n-1)
	ret = append(ret, ss[:i]...)
	ret = append(ret, ss[i+1:]...)
	return ret
}

func randomShrink(ss []string) ([]string, string) {

	n := len(ss)
	if n == 1 {
		return ss[0:0], ss[0]
	}
	i := rand.Intn(n)
	s := ss[i]
	ss[i] = ss[0]
	return ss[1:], s
}

func indexRequest(rs []*http.Request, sep *http.Request) int {

	for i, r := range rs {
		if r == sep {
			return i
		}
	}
	return -1
}
