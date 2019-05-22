package utility

import (
	"testing"

	"github.com/qiniu/db/mgoutil.v3"
	"github.com/qiniu/http/restrpc.v1"
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

	srv, _ := New(Config{
		UseMock:              true,
		ServingHost:          "argus.ava.ai",
		TerrorThreshold:      0.25,
		BjRTerrorThreshold:   []float32{0.83, 0.3},
		PoliticianThreshold:  []float32{0.6, 0.66, 0.72},
		PulpReviewThreshold:  0.89,
		PulpFusionThreshold:  []int{128, 192},
		BluedDetectThreshold: 0.8,
		MongConf: &mgoutil.Config{
			Host: "mongodb://127.0.0.1:27017",
			DB:   "atlab",
		},
	})

	router := restrpc.Router{
		PatternPrefix: "v1",
	}

	transport := mockhttp.NewTransport()
	transport.ListenAndServe(srv.Config.ServingHost, router.Register(srv))

	ctx := httptest.New(t)
	ctx.SetTransport(transport)
	return srv, ctx
}
