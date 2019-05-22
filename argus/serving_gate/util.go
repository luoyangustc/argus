package gate

import (
	"fmt"
	"net/http"
	"strings"
)

var (
	NAMESPACE = "ava"
	SUBSYSTEM = "serving_gate"
)

//----------------------------------------------------------------------------//

func p2s(v *string) string {
	if v == nil {
		return "nil"
	}
	return *v
}

func sp(v string) *string {
	return &v
}

//----------------------------------------------------------------------------//

const (
	_V1 = "/v1/eval/"
)

func parseOP(op string) (string, *string, error) {
	var (
		cmd     string
		version *string
	)
	switch {
	case strings.HasPrefix(op, _V1):
		strs := strings.Split(strings.TrimPrefix(op, _V1), "/")
		if len(strs) == 0 || len(strs[0]) == 0 {
			return cmd, version, ErrBadOP
		}
		cmd = strs[0]
		if len(strs) > 1 {
			version = &strs[1]
		}
		return cmd, version, nil
	default:
		return cmd, version, ErrBadOP
	}
}

//----------------------------------------------------------------------------//

func formatVersion(version *string) string {
	if version == nil {
		return ""
	}
	return *version
}

func formatError(err error) string {
	if err == nil {
		return ""
	}
	return err.Error()
}

//------------------------------------------------------------------/
type _EvalEnv struct {
	Uid   uint32
	Utype uint32
}
type evalTransport struct {
	_EvalEnv
	http.RoundTripper
}

func (t evalTransport) RoundTrip(req *http.Request) (*http.Response, error) {
	req.Header.Set(
		"Authorization",
		fmt.Sprintf("QiniuStub uid=%d&ut=%d", t.Uid, 0), // t.Utype), 特殊设置，避免Argus|Serving同时计量计费
	)
	return t.RoundTripper.RoundTrip(req)
}
