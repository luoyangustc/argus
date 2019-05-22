package ops

import (
	"context"
	"encoding/base64"
	"errors"
	"io/ioutil"
	"net/http"
	"time"

	"github.com/qiniu/rpc.v3"
	"github.com/qiniu/xlog.v1"

	ahttp "qiniu.com/argus/argus/com/http"
	"qiniu.com/argus/video"
	"qiniu.com/argus/video/vframe"
)

func RegisterTerrorClassify() {
	video.RegisterOP("terror_classify",
		func(config video.OPConfig) video.OP { return NewSimpleCutOP(config, EvalTerrorClassify) })

	video.RegisterOP("terror_classify_clip",
		func(config video.OPConfig) video.OP { return iTerrorClassifyClip{OPConfig: config} })
}

type TerrorClassifyResp struct {
	Code    int                  `json:"code"`
	Message string               `json:"message"`
	Result  TerrorClassifyResult `json:"result"`
}

type TerrorClassifyResult struct {
	Confidences []struct {
		Index int     `json:"index"`
		Class string  `json:"class"`
		Score float32 `json:"score"`
	} `json:"confidences"`
}

func (r TerrorClassifyResult) Len() int { return len(r.Confidences) }
func (r TerrorClassifyResult) Parse(i int) (string, float32, bool) {
	return r.Confidences[i].Class, r.Confidences[i].Score, true
}

func EvalTerrorClassify(ctx context.Context, client *rpc.Client, host, uri string) (interface{}, error) {
	var (
		xl   = xlog.FromContextSafe(ctx)
		resp TerrorClassifyResp
	)
	err := client.CallWithJson(ctx, &resp, "POST", host+"/v1/eval/terror-classify",
		struct {
			Data struct {
				URI string `json:"uri"`
			} `json:"data"`
		}{
			Data: struct {
				URI string `json:"uri"`
			}{
				URI: uri,
			},
		},
	)
	if err != nil {
		return nil, err
	}
	if resp.Code != 0 && resp.Code/100 != 2 {
		xl.Warnf("terror classify cut failed. %#v", resp)
		return nil, nil
	}
	return resp.Result, nil
}

////////////////////////////////////////////////////////////////////////////////

type TerrorClassifyClipResult struct { // TODO
	Code    int    `json:"code"`
	Message string `json:"message"`
	Result  struct {
		Segments []struct {
			Label string  `json:"label"`
			Score float32 `json:"score"`
			Start float64 `json:"start"`
			End   float64 `json:"end"`
		} `json:"segments"`
	} `json:"result"`
}

var _ video.OP = iTerrorClassifyClip{}
var _ video.ClipOP = iTerrorClassifyClip{}

type iTerrorClassifyClip struct {
	video.OPConfig
	video.OPEnv
	timeout time.Duration

	Skip       int `json:"skip,omitempty"`        //检测滑动窗口的步长，默认：10
	FrameGroup int `json:"frame_group,omitempty"` //一次用来处理的视频时长,默认:20
}

func (tc iTerrorClassifyClip) Fork(ctx context.Context, params video.OPParams, env video.OPEnv) (video.OP, bool) {
	vc := iTerrorClassifyClip{
		OPConfig: video.OPConfig{
			Host:   tc.OPConfig.Host,
			Params: params,
		},
		OPEnv:      env,
		Skip:       0,
		FrameGroup: 0,
	}
	if params.Other != nil {
		vc.Skip = int(((params.Other).(map[string]interface{})["skip"]).(float64))
		vc.FrameGroup = int(((params.Other).(map[string]interface{})["frame_group"]).(float64))
	} else if tc.Params.Other != nil {
		vc.Skip = int(((tc.Params.Other).(map[string]interface{})["skip"]).(float64))
		vc.FrameGroup = int(((tc.Params.Other).(map[string]interface{})["frame_group"]).(float64))
	}
	return vc, true
}

func (p iTerrorClassifyClip) Reset(context.Context) error { return nil }
func (p iTerrorClassifyClip) Count() int32                { return video.MAX_OP_COUNT }

func (tc iTerrorClassifyClip) NewClips(ctx context.Context, job vframe.Job) video.Clips {
	xl := xlog.FromContextSafe(ctx)
	_, ok := job.(video.VframeJob)
	if !ok {
		xl.Infof("error of vframe job")
		return nil
	}

	xl.Infof("job.FrameGroup: %#v", job.(video.VframeJob).Params())

	var (
		// image -> []feature
		evalFeatureFunc = func(ctx context.Context, uris []string) ([]interface{}, error) {
			var (
				xl     = xlog.FromContextSafe(ctx)
				client = ahttp.NewQiniuStubRPCClient(tc.OPEnv.Uid, tc.OPEnv.Utype, tc.timeout)
				url    = tc.OPConfig.Host + "/v1/eval/video-feature"
				result = make([]interface{}, 0, len(uris))
			)

			xl.Infof("query video-classify feature: %d", len(uris))
			for _, uri := range uris { // TODO 并行
				ret, err := func(uri string) (interface{}, error) {
					xl.Infof("url and data: %#v, %#v", url, video.PrintableURI(uri))

					var resp *http.Response
					f := func(ctx context.Context) error {
						var _e error
						resp, _e = client.DoRequestWithJson(ctx, "POST", url,
							struct {
								Data struct {
									URI string `json:"uri"`
								} `json:"data"`
							}{
								Data: struct {
									URI string `json:"uri"`
								}{
									URI: uri,
								},
							})
						return _e
					}
					err := ahttp.CallWithRetry(
						ctx, []int{530}, []func(context.Context) error{f, f},
					)
					if err != nil {
						xl.Errorf("query video-classify feature failed. error: %v", err)
						return nil, err //失败则返回失败
					}

					defer resp.Body.Close()
					if resp.StatusCode/100 != 2 || resp.ContentLength == 0 {
						xl.Errorf("query video-classify feature failed. status code:%v,content length:%v", resp.StatusCode, resp.ContentLength)
						return nil, err
					}
					bs, err := ioutil.ReadAll(resp.Body)
					if err != nil {
						xl.Errorf("query video-classify feature failed. read resp body error:%v", err)
						return nil, err
					}
					xl.Infof("query video-classify feature result: %d", len(bs))
					return bs, nil
				}(uri)

				if err != nil {
					return nil, err
				}
				result = append(result, ret)
			}

			xl.Infof("query video-classify feature done.")
			return result, nil
		}

		parseFeatureFunc = func(ctx context.Context, _v interface{}) (labels []string, scores []float32, err error) {
			if _v == nil {
				return nil, nil, nil
			}

			v := _v.([]map[string]float32)
			for _, item := range v {
				for key, val := range item {
					labels = append(labels, key)
					scores = append(scores, val)
				}
			}

			return
		}
	)

	clips := video.NewEveryFrameClips(job,
		tc.Skip,
		tc.FrameGroup,
		evalFeatureFunc,
		parseFeatureFunc,
		func(ctx context.Context, features [][]byte) (interface{}, error) {
			type TopNReq struct {
				Data []struct {
					URI string `json:"uri"`
				} `json:"data"`
			}

			type TopNResp struct {
				Code    int                  `json:"code"`
				Message string               `json:"message"`
				Result  []map[string]float32 `json:"result"`
			}

			var (
				xl     = xlog.FromContextSafe(ctx)
				client = ahttp.NewQiniuStubRPCClient(tc.OPEnv.Uid, tc.OPEnv.Utype, tc.timeout)
				url    = tc.OPConfig.Host + "/v1/eval/video-classify"
			)
			xl.Infof("query video-classify: %d", len(features))

			req := TopNReq{
				Data: make([]struct {
					URI string `json:"uri"`
				}, 0, len(features)),
			}
			for _, feature := range features {
				req.Data = append(req.Data, struct {
					URI string `json:"uri"`
				}{
					URI: "data:application/octet-stream;base64," + base64.StdEncoding.EncodeToString(feature),
				})
			}

			xl.Infof("url and data: %#v, %#v", url, req)
			var resp TopNResp
			f := func(ctx context.Context) error {
				return client.CallWithJson(ctx, &resp, "POST", url, req)
			}
			if err := ahttp.CallWithRetry(
				ctx, []int{530}, []func(context.Context) error{f, f},
			); err != nil {
				xl.Errorf("query video-classify failed. error: %v, resp: %#v", err, resp)
				return nil, err //失败则返回失败
			}

			xl.Infof("query video-classify topN done. %#v", resp)

			if resp.Code != 0 {
				xl.Warnf("query video-classify topN failed. %#v", resp)
				return nil, errors.New(resp.Message)
			}

			return resp.Result, nil
		},
	)
	return clips
}
