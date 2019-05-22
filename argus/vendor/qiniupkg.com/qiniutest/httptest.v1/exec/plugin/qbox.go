package plugin

import (
	"net/http"

	"qiniupkg.com/api.v7/auth/qbox"
	"qiniupkg.com/httptest.v1"
)

// ---------------------------------------------------------------------------

type authTransportComposer struct {
	mac *qbox.Mac
}

func (p authTransportComposer) Compose(base http.RoundTripper) http.RoundTripper {
	return qbox.NewTransport(p.mac, base)
}

// ---------------------------------------------------------------------------

type qboxArgs struct {
	AK string `arg:"access-key"`
	SK string `arg:"secret-key"`
}

func (p *subContext) Eval_qbox(ctx *httptest.Context, args *qboxArgs) (httptest.TransportComposer, error) {

	mac := &qbox.Mac{
		AccessKey: args.AK,
		SecretKey: []byte(args.SK),
	}
	return authTransportComposer{mac}, nil
}

// ---------------------------------------------------------------------------

