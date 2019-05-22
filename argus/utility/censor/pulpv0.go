package censor

import (
	"context"
	"fmt"
	"image"
	"math/rand"
	"sync"
	"time"

	"github.com/qiniu/http/httputil.v1"
	"github.com/qiniu/xlog.v1"
	"qiniu.com/argus/argus/com/util"
	"qiniu.com/argus/utility/evals"
	"qiniu.com/argus/utility/server"
	"qiniu.com/auth/authstub.v1"
)

type PulpV0Req struct {
	Images []string `json:"image"`
}

type PulpV0Ret struct { //Ai Ret
	Code      int    `json:"code"`
	Message   string `json:"message"`
	Nonce     string `json:"nonce"`     // random value
	TimeStamp int64  `json:"timestamp"` // unix server time stamp
	Nrop      Pulp   `json:"pulp"`      //the mark for pulp testing
}

type Rec struct {
	Rate   float32 `json:"rate"`
	Label  int     `json:"label"`
	Name   string  `json:"name"`
	Review bool    `json:"review"`
}

type File struct {
	Rec `json:"result"`
}

type Pulp struct {
	ReviewCount int    `json:"reviewCount"`
	Statistic   []int  `json:"statistic"`
	FileList    []File `json:"fileList"`
}

//The POST method provides a compatiable
func (s *Service) PostPulpRecognition(ctx context.Context, args *PulpV0Req, env *authstub.Env) (ret *PulpV0Ret, err error) {

	defer func(begin time.Time) {
		server.ResponseTimeAtServer("pulpv0", "").
			Observe(server.DurationAsFloat64(time.Since(begin)))
		server.HttpRequestsCounter("pulpv0", "", server.FormatError(err)).Inc()
	}(time.Now())

	var (
		ctex, xl   = util.CtxAndLog(ctx, env.W, env.Req)
		waiter     sync.WaitGroup
		lock       sync.Mutex
		cfailed    int
		_5XXErr    int
		_Not5XXErr int
	)

	ret = new(PulpV0Ret)
	ret.Nrop.Statistic = make([]int, 3)
	ret.Nonce = fmt.Sprintf("%d", rand.Int31())
	ret.TimeStamp = time.Now().Unix()

	xl.Infof("request arguments: %v", args)
	if len(args.Images) == 0 || len(args.Images) > 50 {
		return nil, server.ErrArgs
	}

	waiter.Add(len(args.Images))
	for _, img := range args.Images {
		go func(_image string, ctx context.Context) {
			defer waiter.Done()

			var (
				xl    = xlog.FromContextSafe(ctx)
				preq  evals.PulpReq
				eResp PulpResp
				_err  error
			)
			_image, _ = server.ImproveURI(_image, env.Uid)
			img, _err := s.ParseImage(ctx, _image)
			if _err != nil && _err != image.ErrFormat {
				xl.Infof("parse image failed. %v", err)
			} else {
				preq.Data.URI = _image
				if img.URI != nil {
					preq.Data.URI = *img.URI
				}
				preq.Params.Limit = 3
				eResp, _err = s.postPulp(ctx, &preq, img, env.Uid, env.Utype)
			}

			lock.Lock()
			defer lock.Unlock()
			if _err != nil || (eResp.Code != 0 && eResp.Code/100 != 2) || len(eResp.Result.Confidences) == 0 {
				if _err != nil && httputil.DetectCode(_err)/100 == 5 {
					_5XXErr = httputil.DetectCode(_err)
				} else if _err != nil {
					_Not5XXErr = httputil.DetectCode(_err)
				} else {
					_Not5XXErr = 424
				}

				xl.Errorf("ARGUS_PULP_V0 error: %v,aresp: %v", _err, eResp)
				ret.Nrop.FileList = append(ret.Nrop.FileList, File{Rec{
					Name:   _image,
					Label:  -1,
					Review: true,
				}})
				cfailed++
				ret.Nrop.ReviewCount++
				return
			}
			var reivew bool
			if eResp.Result.Confidences[0].Score < 0.6 {
				reivew = true
				ret.Nrop.ReviewCount++
			}
			ret.Nrop.FileList = append(ret.Nrop.FileList, File{Rec{
				Label:  eResp.Result.Confidences[0].Index,
				Rate:   eResp.Result.Confidences[0].Score,
				Name:   _image,
				Review: reivew,
			}})
			ret.Nrop.Statistic[eResp.Result.Confidences[0].Index]++
		}(img, util.SpawnContext(ctex))
	}
	waiter.Wait()

	if cfailed == len(args.Images) {
		if env.UserInfo.Utype != server.NoChargeUtype {
			server.SetStateHeader(env.W.Header(), "PULP-Depend", len(args.Images))
		}
		errCode := 599
		if _5XXErr != 0 {
			errCode = _5XXErr
		} else if _Not5XXErr != 0 {
			errCode = _Not5XXErr
		}
		err = httputil.NewError(errCode, "all image testing failed")
		return nil, err
	} else if cfailed != 0 {
		ret.Message = "partially success"
	} else {
		ret.Message = "success"
	}
	if env.UserInfo.Utype != server.NoChargeUtype {
		server.SetStateHeader(env.W.Header(), "PULP-Certain", len(args.Images)-ret.Nrop.ReviewCount)
		server.SetStateHeader(env.W.Header(), "PULP-Depend", ret.Nrop.ReviewCount)
	}
	return
}
