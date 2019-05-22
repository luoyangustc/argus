package utility

import (
	"context"
	"encoding/base64"
	"fmt"
	"io/ioutil"
	"net/http"
	"sync"
	"time"

	"qiniu.com/argus/utility/evals"

	"github.com/qiniu/http/httputil.v1"
	"github.com/qiniu/rpc.v3"
	"github.com/qiniu/xlog.v1"
	"qiniu.com/argus/atserving/model"
	"qiniu.com/auth/authstub.v1"
)

type _EvalFaceClusterReqSource struct {
	URI string `json:"uri"`
}

type _EvalFaceClusterReq struct {
	Data []_EvalFaceClusterReqSource `json:"data"`
}

type _EvalFaceClusterResp struct {
	Code    int                  `json:"code"`
	Message string               `json:"message"`
	Result  _EvalFaceClusterInfo `json:"result"`
}

type _EvalFaceClusterInfo struct {
	Fcluster []_EvalFaceClusterDetail `json:"facex_cluster"`
}

type _EvalFaceClusterDetail struct {
	ID         int     `json:"cluster_id"`
	CenterDist float32 `json:"cluster_center_dist"`
}

type iFaceFeature interface {
	Eval(context.Context, _EvalFaceReq, _EvalEnv) ([]byte, error)
}

type _FaceFeature struct {
	url     string
	timeout time.Duration
	*rpc.Client
}

func newFaceFeature(host string, timeout time.Duration) _FaceFeature {
	return _FaceFeature{url: host + "/v1/eval/facex-feature", timeout: timeout}
}

func (ff _FaceFeature) Eval(
	ctx context.Context, req _EvalFaceReq, env _EvalEnv,
) (bs []byte, err error) {

	var (
		xl     = xlog.FromContextSafe(ctx)
		client *rpc.Client
	)
	if ff.Client == nil {
		client = newRPCClient(env, ff.timeout)
	} else {
		client = ff.Client
	}
	var resp *http.Response
	err = callRetry(ctx,
		func(ctx context.Context) error {
			var err1 error
			resp, err1 = client.DoRequestWithJson(ctx, "POST", ff.url, &req)
			return err1
		})
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode/100 != 2 || resp.ContentLength == 0 {
		xl.Errorf(
			"call "+ff.url+" error:%v,status code:%v,content length:%v,req:%v",
			err, resp.StatusCode, resp.ContentLength, req,
		)
		return nil, httputil.NewError(http.StatusInternalServerError, "get feature error")
	}
	bs, err = ioutil.ReadAll(resp.Body)
	if err != nil {
		xl.Errorf("call "+ff.url+",read resp body error:%v", err)
	}
	return bs, err
}

type iFaceCluster interface {
	Eval(context.Context, _EvalFaceClusterReq, _EvalEnv) (_EvalFaceClusterResp, error)
}

type _FaceCluster struct {
	host    string
	timeout time.Duration
}

func newFaceCluster(host string, timeout time.Duration) iFaceCluster {
	return _FaceCluster{host: host, timeout: timeout}
}

func (fc _FaceCluster) Eval(
	ctx context.Context, req _EvalFaceClusterReq, env _EvalEnv,
) (resp _EvalFaceClusterResp, err error) {
	var (
		url    = fc.host + "/v1/eval/facex-cluster"
		client = newRPCClient(env, fc.timeout)
	)
	err = callRetry(ctx,
		func(ctx context.Context) error {
			return client.CallWithJson(ctx, &resp, "POST", url, &req)
		})
	return
}

//----------------------------------------------------------------------------//

type FaceClusterResp struct {
	Code    int               `json:"code"`
	Message string            `json:"message"`
	Result  FaceClusterResult `json:"result"`
}

type FaceClusterResult struct {
	Cluster [][]FaceClusterDetail `json:"cluster"`
}

type FaceClusterDetail struct {
	BoundingBox FaceDetectBox    `json:"boundingBox"`
	Group       FaceClusterGroup `json:"group"`
}

type FaceClusterGroup struct {
	ID         int     `json:"id"`
	CenterDist float32 `json:"center_dist"`
}

func (s *Service) PostFaceCluster(ctx context.Context, args *model.GroupEvalRequest, env *authstub.Env) (ret *FaceClusterResp, err error) {

	ctx, xl := ctxAndLog(ctx, env.W, env.Req)
	var (
		uid   = env.UserInfo.Uid
		utype = env.UserInfo.Utype
	)

	xl.Infof("PostFaceCluster Req Arguments: %v", args)
	if len(args.Data) < 2 {
		return nil, ErrArgs
	}

	var (
		featureProc = make(chan map[string]interface{}, 15)
		imgPts      = make(map[string][]FaceDetectBox)
		ftSrc       = make(map[string][]_EvalFaceClusterReqSource)

		fclusterReq  _EvalFaceClusterReq
		fclusterResp _EvalFaceClusterResp

		evalEnv = _EvalEnv{Uid: uid, Utype: utype}
	)
	ret = new(FaceClusterResp)

	{
		fwaiter := conFeatureFetch(ctx, s.facexFeatureV3, featureProc, imgPts, ftSrc, evalEnv)
		var waiter sync.WaitGroup
		waiter.Add(len(args.Data))
		for _, src := range args.Data {
			go func(ctx context.Context, image string) {
				xl := xlog.FromContextSafe(ctx)
				defer waiter.Done()
				var fdtReq _EvalFaceDetectReq
				fdtReq.Data.URI = image
				fdtResp, inerr := s.iFaceDetect.Eval(ctx, fdtReq, _EvalEnv{Uid: uid, Utype: utype})
				if inerr != nil {
					xl.Errorf("call "+s.Config.ServingHost+"/v1/eval/facex-detect error:%v", inerr)
					return
				}
				if fdtResp.Code != 0 && fdtResp.Code/100 != 2 {
					xl.Errorf("call "+s.Config.ServingHost+"/v1/eval/facex-detect, code is not equal to zero:%v", fdtResp)
					return
				}

				for _, dt := range fdtResp.Result.Detections {
					fc := make(map[string]interface{})
					fc["uri"] = image
					fc["pts"] = dt.Pts
					fc["score"] = dt.Score
					featureProc <- fc
				}

			}(spawnContext(ctx), src.URI.String())
		}
		waiter.Wait()
		close(featureProc)
		fwaiter.Wait()
	}

	{
		xl.Infof("\nimage pts info %v\n", imgPts)
		xl.Infof("\nimage feature info %v\n", len(ftSrc))
		for _, src := range args.Data {
			if lsrc, ok := ftSrc[src.URI.String()]; ok {
				fclusterReq.Data = append(fclusterReq.Data, lsrc...)
			}
		}

		if len(fclusterReq.Data) < 2 {
			err = httputil.NewError(http.StatusBadRequest, fmt.Sprintf("no enought face detect within the images,face num:%v", len(fclusterReq.Data)))
			return
		}

		fclusterResp, err = s.iFaceCluster.Eval(ctx, fclusterReq, evalEnv)
		if err != nil || (fclusterResp.Code != 0 && fclusterResp.Code/100 != 2) {
			xl.Errorf("call "+s.Config.ServingHost+"/v1/eval/facex-cluster error:%v,resp:%v", err, fclusterResp)
			return nil, err
		}
		xl.Infof("facex-cluster resp:%v", fclusterResp)
	}

	{
		count := 0
		for _, src := range args.Data {
			if _, ok := imgPts[src.URI.String()]; !ok {
				ret.Result.Cluster = append(ret.Result.Cluster, make([]FaceClusterDetail, 0))
				continue
			}
			signleImg := make([]FaceClusterDetail, 0)
			for i, b := range imgPts[src.URI.String()] {
				signleImg = append(signleImg, FaceClusterDetail{
					BoundingBox: b,
					Group: FaceClusterGroup{
						ID:         fclusterResp.Result.Fcluster[count+i].ID,
						CenterDist: fclusterResp.Result.Fcluster[count+i].CenterDist,
					},
				})
			}
			count += len(imgPts[src.URI.String()])
			ret.Result.Cluster = append(ret.Result.Cluster, signleImg)
		}
	}
	return
}

func conFeatureFetch(
	ctx context.Context,
	fFeature evals.IFaceFeature,
	featureProc chan map[string]interface{},
	imgPts map[string][]FaceDetectBox,
	ftSrc map[string][]_EvalFaceClusterReqSource,
	env _EvalEnv,
) *sync.WaitGroup {

	// xl := xlog.FromContextSafe(ctx)
	m := new(sync.Mutex)
	var fwaiter sync.WaitGroup
	fwaiter.Add(15)
	for i := 0; i < 15; i++ {
		go func(ctx context.Context) {
			xl := xlog.FromContextSafe(ctx)
			defer fwaiter.Done()
			for {
				p, ok := <-featureProc
				if !ok {
					break
				}
				fftReq := evals.FaceReq{}
				fftReq.Data.URI = p["uri"].(string)
				fftReq.Data.Attribute.Pts = p["pts"].([][2]int)

				fft, err := fFeature.Eval(ctx, fftReq, env.Uid, env.Utype)
				if err != nil {
					xl.Errorf("call feature: %v", err)
					continue
				}

				b64fft := base64.StdEncoding.EncodeToString(fft)
				xl.Infof("featuer encoding info,uri:%v,len(b64fft):%v,len(fft):%v", p["uri"], len(b64fft), len(fft))
				m.Lock()
				imgPts[p["uri"].(string)] = append(
					imgPts[p["uri"].(string)],
					FaceDetectBox{
						Pts:   p["pts"].([][2]int),
						Score: p["score"].(float32),
					},
				)
				ftSrc[p["uri"].(string)] = append(
					ftSrc[p["uri"].(string)],
					_EvalFaceClusterReqSource{
						URI: "data:application/octet-stream;base64," + b64fft,
					},
				)
				m.Unlock()
			}
		}(spawnContext(ctx))
	}
	return &fwaiter
}
