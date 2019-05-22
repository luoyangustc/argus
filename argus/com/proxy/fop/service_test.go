package fop

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"io/ioutil"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/stretchr/testify/assert"

	"github.com/qiniu/http/restrpc.v1"
	"github.com/qiniu/http/servestk.v1"
)

func TestService(t *testing.T) {

	ts1 := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		fmt.Fprintln(w, "Hello, client")
	}))
	defer ts1.Close()

	mux := servestk.New(restrpc.NewServeMux())
	router := &restrpc.Router{
		PatternPrefix: "v1",
		Factory:       restrpc.Factory,
		Mux:           mux,
	}

	ts2 := httptest.NewServer(mux)
	defer ts2.Close()

	proxy := &mockProxy{}
	router.Register(NewService(ts2.URL+"/v1", proxy))

	{
		url := fmt.Sprintf("%s/v1/handler?cmd=foo&url=%s", ts2.URL, ts1.URL)
		req, _ := http.NewRequest("POST", url, nil)
		req.Header.Set("X-Qiniu-Uid", base64.StdEncoding.EncodeToString([]byte("12345678")))
		resp, err := http.DefaultClient.Do(req)
		assert.NoError(t, err)
		defer resp.Body.Close()
		var v = struct {
			Len int `json:"len"`
		}{}
		assert.NoError(t, json.NewDecoder(resp.Body).Decode(&v))
		assert.Equal(t, 14, v.Len)
		assert.Equal(t, uint32(12345678), proxy.req.UID)
	}

}

//----------------------------------------------------------------------------//

var _ Proxy = &mockProxy{}

type mockProxy struct {
	req ProxyReq
}

func (mock *mockProxy) Post(ctx context.Context, req ProxyReq, env *restrpc.Env) (interface{}, error) {
	mock.req = req
	var body io.Reader
	if req.URL == "" {
		body = env.Req.Body
	} else {
		fmt.Println(req.URL)
		resp, err := http.Get(req.URL)
		if err != nil {
			return nil, err
		}
		defer resp.Body.Close()
		body = resp.Body
	}
	bs, err := ioutil.ReadAll(body)
	if err != nil {
		return nil, err
	}
	fmt.Println(string(bs))
	return struct {
		Len int `json:"len"`
	}{
		Len: len(bs),
	}, nil
}
