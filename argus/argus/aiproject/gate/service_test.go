package gate

import (
	"context"
	"github.com/stretchr/testify.v2/assert"
	"net/http"
	"reflect"
	"testing"

	"github.com/qiniu/http/rpcutil.v1"
	"qiniupkg.com/qiniutest/httptest.v1"
	"qiniupkg.com/x/mockhttp.v7"
	"qiniupkg.com/x/xlog.v7"
)

type tLog struct {
	t *testing.T
}

func (t *tLog) Write(p []byte) (n int, err error) {
	t.t.Log(string(p))
	return len(p), nil
}

func getMockContext(t *testing.T) (*Service, httptest.Context) {
	xlog.SetOutputLevel(0)
	xlog.SetOutput(&tLog{t: t})

	mux := http.NewServeMux()
	srv, _ := New(nil)
	method, _ := reflect.TypeOf(srv).MethodByName("Do")
	handler, _ := rpcutil.HandlerCreator{}.New(reflect.ValueOf(srv), method)
	mux.Handle("/", handler)

	transport := mockhttp.NewTransport()
	transport.ListenAndServe("aiproject.ava.ai", mux)

	ctx := httptest.New(t)
	ctx.SetTransport(transport)
	return srv, ctx
}

type mockRouter struct {
}

func (mr mockRouter) Match(ctx context.Context, app string) string {
	return ""
}

func TestService(t *testing.T) {

	cmd, path, err := parseUrl("app/custom/test?kk=5")
	assert.Nil(t, err)
	assert.Equal(t, "app", cmd)
	assert.Equal(t, "/custom/test?kk=5", path)

	service, ctx := getMockContext(t)
	service.router = mockRouter{}

	ctx.Exec(`
		post http://aiproject.ava.ai/test/reverse
		auth |authstub -uid 1 -utype 4|
		json '{
			"data": {
				"uri": "http://test.image.jpg"  
			}   
		}'
		ret 404
	`)
}
