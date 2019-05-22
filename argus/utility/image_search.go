package utility

import (
	"context"
	"encoding/base64"
	"strings"
	"time"

	"qiniu.com/auth/authstub.v1"
)

type _EvalImageSearchReq struct {
	Data struct {
		URI string `json:"uri"`
	} `json:"data"`
	Params *struct {
		Limit *int `json:"limit,omitempty"`
	} `json:"params,omitempty"`
}

type _EvalImageSearchResp struct {
	Code    int    `json:"code"`
	Message string `json:"message"`
	Result  []struct {
		Class string  `json:"class"`
		Label string  `json:"label"`
		Score float32 `json:"score"`
	} `json:"result"`
}

type iImageSearch interface {
	Eval(context.Context, string, _EvalImageSearchReq, _EvalEnv) (_EvalImageSearchResp, error)
}

type _ImageSearch struct {
	host    string
	timeout time.Duration
}

func newImageSearch(host string, timeout time.Duration) iImageSearch {
	return _ImageSearch{host: host, timeout: timeout}
}

func (isb _ImageSearch) Eval(
	ctx context.Context, name string, req _EvalImageSearchReq, env _EvalEnv,
) (_EvalImageSearchResp, error) {
	var (
		url    = isb.host + "/v1/eval/search-" + name
		client = newRPCClient(env, isb.timeout)

		resp _EvalImageSearchResp
	)
	err := callRetry(ctx,
		func(ctx context.Context) error {
			return client.CallWithJson(ctx, &resp, "POST", url, &req)
		})
	return resp, err
}

//----------------------------------------------------------------------------//

// ImageSearchReq ...
type ImageSearchReq struct {
	CmdArgs []string
	Data    struct {
		URI string `json:"uri"`
	} `json:"data"`
	Params struct {
		Limit int `json:"limit,omitempty"`
	} `json:"params,omitempty"`
}

// ImageSearchResp ...
type ImageSearchResp struct {
	Code    int                 `json:"code"`
	Message string              `json:"message"`
	Result  []ImageSearchResult `json:"result"`
}

// ImageSearchResult ...
type ImageSearchResult struct {
	URL   string  `json:"url"`
	Label string  `json:"label"`
	Score float32 `json:"score"`
}

// PostImageSearch_ ...
func (s *Service) PostImageSearch_(
	ctx context.Context, args *ImageSearchReq, env *authstub.Env,
) (ret *ImageSearchResp, err error) {

	var (
		uid     = env.UserInfo.Uid
		utype   = env.UserInfo.Utype
		evalEnv = _EvalEnv{Uid: uid, Utype: utype}

		name = args.CmdArgs[0]
	)
	ctx, xl := ctxAndLog(ctx, env.W, env.Req)

	if strings.TrimSpace(args.Data.URI) == "" {
		xl.Error("empty data.uri")
		return nil, ErrArgs
	}

	var fReq _EvalImageReq
	fReq.Data.URI = args.Data.URI
	f, err := s.iFeatureV2.Eval(ctx, fReq, evalEnv)
	if err != nil {
		xl.Errorf("get image feature failed. %v", err)
		return
	}
	if len(f) < 5 {
		xl.Infof("query feature error: %v", err)
		return nil, ErrArgs
	}

	xl.Infof("feature: %d", len(f))

	var mReq _EvalImageSearchReq
	mReq.Data.URI = "data:application/octet-stream;base64," +
		base64.StdEncoding.EncodeToString(f)
	if args.Params.Limit == 0 {
		args.Params.Limit = 5
	}
	mReq.Params = &struct {
		Limit *int `json:"limit,omitempty"`
	}{Limit: &args.Params.Limit}
	mResp, err := s.iImageSearch.Eval(ctx, name, mReq, evalEnv)
	if err != nil {
		xl.Errorf("get image search failed. %v", err)
		return
	}

	ret = &ImageSearchResp{
		Code:    mResp.Code,
		Message: mResp.Message,
		Result:  make([]ImageSearchResult, 0, len(mResp.Result)),
	}
	for _, result := range mResp.Result {
		ret.Result = append(ret.Result,
			ImageSearchResult{URL: result.Class, Score: result.Score, Label: result.Label})
	}

	return
}
