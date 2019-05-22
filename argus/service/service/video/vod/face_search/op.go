package face_search

import (
	"context"
	"encoding/json"
	"errors"

	"qiniu.com/argus/video/vframe"

	"github.com/go-kit/kit/endpoint"
	feature_group "qiniu.com/argus/feature_group_private"
	"qiniu.com/argus/sdk/video"
	. "qiniu.com/argus/service/service"
	"qiniu.com/argus/service/service/image"
	svideo "qiniu.com/argus/service/service/video"
	"qiniu.com/argus/service/service/video/vod"
	video0 "qiniu.com/argus/video"
)

func NewOP(gen func() endpoint.Endpoint) svideo.OPFactory {

	return vod.SimpleCutOPFactory{
		NewCutsFunc: func(
			ctx context.Context,
			opParams video0.OPParams,
			_ vframe.VframeParams,
			options ...video.CutOpOption) (video.CutsPipe, error) {

			eval := gen()
			params_, err := ParseParams(opParams)
			if err != nil {
				return nil, ErrArgs(err.Error())
			}

			params := params_.(struct {
				Groups    []string `json:"groups"`
				Cluster   string   `json:"cluster"`
				Threshold float32  `json:"threshold"`
				Limit     int      `json:"limit"`
			})

			func_ := func(ctx context.Context, cut *video.Cut) (interface{}, error) {
				req := FaceSearchReq{}
				body, _ := cut.Body()
				req.Data.IMG.URI = image.DataURI(body)
				req.Params = params
				resp, err := FaceSearchOP{}.DoEval(ctx, req, eval)
				if err != nil {
					return resp, err
				}

				ret := video0.CutResultWithLabels{
					CutResult: video0.CutResult{
						Result: resp,
					},
				}
				ret.Labels, _ = Parse(ctx, resp)
				return ret, nil
			}
			return video.CreateCutOP(func_, options...)
		},
	}
}

func Parse(ctx context.Context, r FaceGroupPrivateSearchResult) ([]video0.ResultLabel, error) {
	ret := make([]video0.ResultLabel, 0)

	for _, v := range r.Faces {
		if len(v.Faces) > 0 {
			ret = append(ret, video0.ResultLabel{
				Name:  string(v.Faces[0].Tag),
				Score: v.Faces[0].Score,
			})
		}
	}
	return ret, nil
}

func ParseParams(params video0.OPParams) (interface{}, error) {
	var req0 FaceSearchReq
	{
		bs, _ := json.Marshal(params.Other)
		err := json.Unmarshal(bs, &req0.Params)
		if err != nil {
			return nil, err
		}
		if req0.Params.Threshold < 0 || req0.Params.Threshold > 1 {
			return nil, errors.New("op params: invalid threshold, should be [0,1]")
		}
		// limit in [1,20]，default limit = 1, 上线20出于H264的SEI长度限制考虑
		if req0.Params.Limit < 0 || req0.Params.Limit > 20 {
			return nil, errors.New("op params: invalid limit, should be [1,20]")
		}
		if req0.Params.Limit == 0 {
			req0.Params.Limit = 1
		}

		// 排重库和聚类库不能同时为空
		if len(req0.Params.Groups) == 0 && len(req0.Params.Cluster) == 0 {
			return nil, errors.New("op params: groups and cluster are both empty")
		}
		for _, group := range req0.Params.Groups {
			if len(group) == 0 {
				return nil, errors.New("op params: group id is empty")
			}
		}
	}
	return req0.Params, nil
}

type FaceSearchReq struct {
	Data struct {
		IMG image.Image
	} `json:"data"`
	Params struct {
		Groups    []string `json:"groups"`
		Cluster   string   `json:"cluster"`
		Threshold float32  `json:"threshold"`
		Limit     int      `json:"limit"`
	} `json:"params"`
}

type FaceSearchResp struct {
	Code    int                            `json:"code"`
	Message string                         `json:"message"`
	Results []FaceGroupPrivateSearchResult `json:"search_results"`
}

type FaceGroupPrivateSearchResult struct {
	Faces []feature_group.FaceSearchRespItem `json:"faces"`
}

type FaceSearchService interface {
	FaceSearch(ctx context.Context, img FaceSearchReq) (FaceSearchResp, error)
}

type FaceSearchOP struct {
	FaceSearchService
}

func NewFaceSearchOP(s FaceSearchService) FaceSearchOP {
	return FaceSearchOP{FaceSearchService: s}
}
func (op FaceSearchOP) NewEval() endpoint.Endpoint {
	return func(ctx context.Context, req_ interface{}) (interface{}, error) {
		req := req_.(FaceSearchReq)
		return op.eval(ctx, req)
	}
}
func (op FaceSearchOP) DoEval(
	ctx context.Context, req FaceSearchReq, ep endpoint.Endpoint,
) (FaceGroupPrivateSearchResult, error) {
	resp, err := ep(ctx, req)
	if resp != nil {
		return resp.(FaceGroupPrivateSearchResult), err
	}
	return FaceGroupPrivateSearchResult{}, err
}
func (op FaceSearchOP) eval(ctx context.Context, req FaceSearchReq) (FaceGroupPrivateSearchResult, error) {
	resp, err := op.FaceSearchService.FaceSearch(ctx, req)
	if err != nil {
		return FaceGroupPrivateSearchResult{}, err
	}
	if len(resp.Results) == 0 {
		return FaceGroupPrivateSearchResult{}, nil
	}
	return resp.Results[0], nil
}
