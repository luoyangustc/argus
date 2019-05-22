package gate

import (
	"context"
	"net/http"

	xlog "github.com/qiniu/xlog.v1"
	"qbox.us/net/httputil"
	. "qiniu.com/argus/service/service"
	authstub "qiniu.com/auth/authstub.v1"
)

const (
	MIME_IMAGE = "image/"
	MIME_VIDEO = "video/"
	MIME_AUDIO = "audio/"
)

type JsonRequest struct {
	CmdArgs []string
	Data    struct {
		URI string `json:"uri"`
	} `json:"data"`
	Params struct {
		Type string `json:"type"`
	} `json:"params"`
}

type Server interface {
	PostWangan(context.Context, *JsonRequest, *authstub.Env)
}

var _ Server = &server{}

type server struct {
	Gate
}

func NewServer(gate Gate) Server {
	return &server{
		Gate: gate,
	}
}

func (s *server) PostWangan(ctx context.Context, req *JsonRequest, env *authstub.Env) {
	var (
		xl   = xlog.FromContextSafe(ctx)
		resp interface{}
		err  error
	)

	resp, err = s.Gate.Call(ctx, *req)
	if err != nil {
		xl.Warnf("PostWangan: call wangan mix failed: %s", err)
		var (
			httpCode int
			msg      string
		)
		info, ok := err.(DetectErrorer)
		if ok {
			httpCode, _, msg = info.DetectError()
		} else {
			httpCode, msg = httputil.DetectError(err)
		}
		httputil.ReplyErr(env.W, httpCode, msg)
		return
	}

	xl.Debugf("wangan resp json: %#v", resp)
	httputil.Reply(env.W, http.StatusOK, resp)
	return
}
