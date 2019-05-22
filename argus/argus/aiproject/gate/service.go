package gate

import (
	"context"
	"fmt"
	"net/http"
	"net/http/httputil"
	"strings"

	"github.com/qiniu/errors"
	qhttputil "github.com/qiniu/http/httputil.v1"
	"qiniu.com/auth/authstub.v1"
	"qiniu.com/argus/argus/com/util"
)

type Service struct {
	router Router
}

func New(rt Router) (*Service, error) {
	return &Service{
		router: rt,
	}, nil
}

func (s Service) Do(ctx context.Context, env *authstub.Env) {

	var (
		ctex, xl = util.CtxAndLog(ctx, env.W, env.Req)
	)

	cmd, path, err := parseUrl(env.Req.URL.Path)
	if err != nil {
		xl.Error("missing app name")
		qhttputil.ReplyErr(env.W, http.StatusNotAcceptable, err.Error())
		return
	}

	host := s.router.Match(ctex, cmd)
	if strings.TrimSpace(host) == "" {
		xl.Errorf("query microService error,cmd :%v, path:%v, env.Req.RequestURI:%v, no valid path", cmd, path, env.Req.RequestURI)
		qhttputil.ReplyErr(env.W, http.StatusNotFound, "no available service found")
		return
	}

	proxy := httputil.ReverseProxy{
		Director: func(req *http.Request) {
			req.URL.Scheme = "http"
			req.URL.Host = host
			req.URL.Path = path
			req.Header.Set("X-Qiniu-Uid", fmt.Sprintf("%d", env.Uid))

			req.Host = host
			req.RequestURI = path
		},
	}

	proxy.ModifyResponse = func(resp *http.Response) error {
		if measure := resp.Header.Get("X-Origin-A"); measure == "" {
			if resp.StatusCode/100 == 5 {
				resp.Header.Set("X-Origin-A", "AIPROJECT_AUTO_"+strings.ToUpper(cmd)+":0")
			} else {
				resp.Header.Set("X-Origin-A", "AIPROJECT_AUTO_"+strings.ToUpper(cmd)+":1")
			}
		} else {
			splits := strings.Split(measure, ";")
			for i, _ := range splits {
				splits[i] = "AIPROJECT_" + strings.ToUpper(cmd) + "_" + splits[i]
			}
			if resp.StatusCode/100 == 5 {
				resp.Header.Set("X-Origin-A", strings.Join(splits, ";")+";"+"AIPROJECT_AUTO_"+strings.ToUpper(cmd)+":0")
			} else {
				resp.Header.Set("X-Origin-A", strings.Join(splits, ";")+";"+"AIPROJECT_AUTO_"+strings.ToUpper(cmd)+":1")
			}
		}

		return nil
	}

	proxy.ServeHTTP(env.W, env.Req)
}

func parseUrl(rawUrl string) (cmd, path string, err error) {
	rawUrl = strings.TrimLeft(rawUrl, "/")
	splits := strings.SplitN(rawUrl, "/", 2)
	if len(splits) == 0 {
		return "", "", errors.New("cmd is required")
	}
	cmd = splits[0]
	if len(splits) > 1 {
		path = "/" + splits[1]
	}
	return
}
