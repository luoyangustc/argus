package job

import (
	"bytes"
	"context"
	"net/http"
	"time"

	"github.com/qiniu/rpc.v3"
	xlog "github.com/qiniu/xlog.v1"
)

func JobHook(ctx context.Context,
	url string, result []byte, err error,
	minWait, waitDuration time.Duration,
) error {

	var (
		xl   = xlog.FromContextSafe(ctx)
		cli  = &rpc.Client{Client: &http.Client{Timeout: time.Second * 3}}
		d    = minWait
		_err error
	)

	xl.Infof("JobHook %s, %s, %v", url, string(result), err)

	for {
		if err != nil {
			_err = cli.CallWithJson(ctx, nil, "POST", url,
				struct {
					Error string `json:"error"`
				}{Error: err.Error()},
			)
		} else {
			buf := bytes.NewReader(result)
			_err = cli.CallWith(ctx, nil, "POST", url, "application/json", buf, buf.Len())
		}
		if _err == nil {
			return nil
		}
		if d > waitDuration {
			break
		}
		time.Sleep(d)
		d *= 2
	}
	return _err
}
