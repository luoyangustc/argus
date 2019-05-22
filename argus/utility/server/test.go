package server

import (
	"context"
	"testing"

	"github.com/qiniu/http/restrpc.v1"
	"github.com/qiniu/xlog.v1"
	"qiniupkg.com/qiniutest/httptest.v1"
	"qiniupkg.com/x/mockhttp.v7"
)

type tLog struct {
	t *testing.T
}

func (t *tLog) Write(p []byte) (n int, err error) {
	t.t.Log(string(p))
	return len(p), nil
}

func NewHTContext(t *testing.T, srv interface{}) httptest.Context {
	xlog.SetOutputLevel(1)
	xlog.SetOutput(&tLog{t: t})

	router := restrpc.Router{PatternPrefix: "v1"}
	transport := mockhttp.NewTransport()
	transport.ListenAndServe("test.com", router.Register(srv))

	ctx := httptest.New(t)
	ctx.SetTransport(transport)
	return ctx
}

////////////////////////////////////////////////////////////////////////////////

type MockStaticServer struct {
	GetEvalF    func(string) interface{}
	ParseImageF func(context.Context, string) (Image, error)
}

func (mock MockStaticServer) GetEval(name string) interface{} { return mock.GetEvalF(name) }
func (mock MockStaticServer) ParseImage(ctx context.Context, uri string) (Image, error) {
	return mock.ParseImageF(ctx, uri)
}
