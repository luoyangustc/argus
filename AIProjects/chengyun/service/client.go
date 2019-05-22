package service

import (
	"context"
	"io"
	"net/http"
	"os"
	"time"

	httputil "github.com/qiniu/http/httputil.v1"
	"github.com/qiniu/rpc.v3"
)

type EvalConfig struct {
	Host string `json:"host"`
	URL  string `json:"url"`
	// timeout: second
	Timeout time.Duration `json:"timeout"`
}

func NewClient(timeout time.Duration) *rpc.Client {
	return &rpc.Client{
		Client: &http.Client{
			Timeout: timeout,
		},
	}
}

//----------------------------------------------------------------------------//

func CallWithRetry(
	ctx context.Context, skipCodes []int,
	calls []func(context.Context) error,
) (err error) {
	for _, call := range calls {

		err = call(ctx)
		if err == nil {
			return
		}

		var unretry = false
		code, _ := httputil.DetectError(err)
		for _, _code := range skipCodes {
			if code == _code {
				unretry = true
				break
			}
		}
		if unretry {
			return
		}
	}
	return
}

func callRetry(ctx context.Context, f func(context.Context) error) error {
	return CallWithRetry(ctx, []int{530}, []func(context.Context) error{f, f})
}

func download(uri, filepath string) (err error) {
	resp, err := http.Get(uri)
	if err != nil {
		return err
	}
	f, err := os.Create(filepath)
	if err != nil {
		return err
	}
	defer f.Close()
	io.Copy(f, resp.Body)
	return
}
