package utility

import (
	"context"
	"time"
)

const (
	HahuiGroupID = "hahui_faces"
)

// --------------------------------------------------------------------------- //
type iHhFaceSearchStatic interface {
	Eval(context.Context, _EvalFaceSearchReq, _EvalEnv) (_EvalFaceSearchResp, error)
}
type _FaceSearchStatic struct {
	host    string
	timeout time.Duration
}

func newHhFaceSearchStatic(host string, timeout time.Duration) _FaceSearchStatic {
	return _FaceSearchStatic{host: host, timeout: timeout}
}

func (fm _FaceSearchStatic) Eval(
	ctx context.Context, req _EvalFaceSearchReq, env _EvalEnv,
) (_EvalFaceSearchResp, error) {
	var (
		url    = fm.host + "/v1/eval/hh-search-static"
		client = newRPCClient(env, fm.timeout)

		resp _EvalFaceSearchResp
	)
	err := callRetry(ctx,
		func(ctx context.Context) error {
			return client.CallWithJson(ctx, &resp, "POST", url, &req)
		})
	return resp, err
}
