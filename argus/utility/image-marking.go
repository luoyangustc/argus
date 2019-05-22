package utility

import (
	"context"
	"math"
	"net/http"
	"sort"
	"strings"
	"sync"
	"time"

	"qbox.us/net/httputil"
	"qiniu.com/auth/authstub.v1"
)

type _EvalObjectClassifyReq struct {
	Data struct {
		URI string `json:"uri"`
	} `json:"data"`
	Params struct {
		Threshold float32 `json:"threshold,omitempty"`
		Limit     int     `json:"limit,omitempty"`
	} `json:"params,omitempty"`
}

type _EvalObjectClassifyResp struct {
	Code    int    `json:"code"`
	Message string `json:"message"`
	Result  struct {
		Confidences []struct {
			Index   int     `json:"index"`
			Class   string  `json:"class"`
			Score   float32 `json:"score"`
			LabelCN string  `json:"label_cn,omitempty"`
		} `json:"confidences"`
	} `json:"result"`
}

type iObjectClassify interface {
	Eval(context.Context, _EvalObjectClassifyReq, _EvalEnv) (_EvalObjectClassifyResp, error)
}

type _ObjectClassify struct {
	host    string
	timeout time.Duration
}

func (ap _ObjectClassify) Eval(ctx context.Context, req _EvalObjectClassifyReq, env _EvalEnv) (_EvalObjectClassifyResp, error) {
	var (
		ret    _EvalObjectClassifyResp
		client = newRPCClient(env, ap.timeout)
	)
	err := callRetry(ctx,
		func(ctx context.Context) error {
			return client.CallWithJson(ctx, &ret, "POST", ap.host+"/v1/eval/object-classify", req)
		})
	return ret, err
}

func newObjectClassify(host string, timeout time.Duration) iObjectClassify {
	return _ObjectClassify{host: host, timeout: timeout}
}

//--------------------------------------------------------------------------------------//

type MarkingReq struct {
	Data struct {
		URI string `json:"uri"`
	} `json:"data"`
}

type MarkingResp struct {
	Code    int           `json:"code"`
	Message string        `json:"message"`
	Result  MarkingResult `json:"result"`
}

type MarkingResult struct {
	Confidences []MarkingObject `json:"confidences"`
}

type MarkingObject struct {
	Class   string  `json:"class"`
	Score   float32 `json:"score"`
	LabelCN string  `json:"label_cn,omitempty"`
}

func (mr *MarkingResult) Sort() {
	sort.Sort(mr)
}

func (mr *MarkingResult) Len() int {
	return len(mr.Confidences)
}
func (mr *MarkingResult) Less(i, j int) bool {
	return mr.Confidences[i].Score > mr.Confidences[j].Score
}
func (mr *MarkingResult) Swap(i, j int) {
	temp := mr.Confidences[i]
	mr.Confidences[i] = mr.Confidences[j]
	mr.Confidences[j] = temp

}

func (s *Service) PostImageLabel(ctx context.Context, args *MarkingReq, env *authstub.Env) (ret *MarkingResp, err error) {

	defer func(begin time.Time) {
		responseTimeAtServer("marking", "").
			Observe(durationAsFloat64(time.Since(begin)))
		httpRequestsCounter("marking", "", formatError(err)).Inc()
	}(time.Now())

	var (
		ctex, xl = ctxAndLog(ctx, env.W, env.Req)
		evalEnv  = _EvalEnv{Uid: env.UserInfo.Uid, Utype: env.UserInfo.Utype}
		set      = make(map[string]MarkingObject)
		waiter   sync.WaitGroup
		mutex    sync.Mutex
	)

	xl.Infof("marking args:%v", args)
	if strings.TrimSpace(args.Data.URI) == "" {
		err = ErrArgs
		return
	}

	waiter.Add(3)
	go func(ctx context.Context) {
		defer waiter.Done()

		var req _EvalSceneReq
		req.Data.URI = args.Data.URI
		req.Params.Limit = 0
		req.Params.Threshold = 0.1
		resp, _err := s.iScene.Eval(ctx, req, evalEnv)
		if _err != nil {
			xl.Errorf("PostMarking query scene error:%v ", _err)
			return
		}
		mutex.Lock()
		defer mutex.Unlock()
		for _, cf := range resp.Result.Confidences {
			nscore := float32(math.Pow(float64(cf.Score), 1.0/3.0))
			nclass := cf.Class
			if v, ok := set[nclass]; ok {
				if nscore < v.Score {
					continue
				}
			}

			set[nclass] = MarkingObject{
				Class:   nclass,
				Score:   nscore,
				LabelCN: cf.LabelCN,
			}
		}
	}(spawnContext(ctex))

	go func(ctx context.Context) {
		defer waiter.Done()

		var req _EvalDetectionReq
		req.Data.URI = args.Data.URI
		req.Params.Threshold = 0.1
		resp, _err := s.iDetection.Eval(ctx, req, evalEnv)
		if _err != nil {
			xl.Errorf("PostMarking query detection error:%v ", _err)
			return
		}
		mutex.Lock()
		defer mutex.Unlock()
		for _, cf := range resp.Result.Detections {
			nscore := float32(math.Pow(float64(cf.Score*0.6), 1.0/3.0))
			if v, ok := set[cf.Class]; ok {
				if nscore < v.Score {
					continue
				}
			}

			set[cf.Class] = MarkingObject{
				Class:   cf.Class,
				Score:   nscore,
				LabelCN: cf.LabelCN,
			}
		}
	}(spawnContext(ctex))

	go func(ctx context.Context) {
		defer waiter.Done()

		var req _EvalObjectClassifyReq
		req.Data.URI = args.Data.URI
		req.Params.Limit = 0
		req.Params.Threshold = 0.05
		resp, _err := s.iObjectClassify.Eval(ctx, req, evalEnv)
		if _err != nil {
			xl.Errorf("PostMarking query object classify error:%v ", _err)
			return
		}
		mutex.Lock()
		defer mutex.Unlock()
		for _, cf := range resp.Result.Confidences {
			nscore := float32(math.Pow(float64(cf.Score), 1.0/3.0))
			if v, ok := set[cf.Class]; ok {
				if nscore < v.Score {
					continue
				}
			}

			set[cf.Class] = MarkingObject{
				Class:   cf.Class,
				Score:   nscore,
				LabelCN: cf.LabelCN,
			}
		}
	}(spawnContext(ctex))

	waiter.Wait()
	if len(set) == 0 {
		err = httputil.NewError(http.StatusInternalServerError, "no valid results")
		return
	}

	ret = new(MarkingResp)
	ret.Result.Confidences = make([]MarkingObject, 0)
	for _, v := range set {
		ret.Result.Confidences = append(ret.Result.Confidences, v)
	}
	ret.Result.Sort()

	if env.UserInfo.Utype != NoChargeUtype {
		setStateHeader(env.W.Header(), "IMAGE-LABEL", 1)
	}
	return
}
