package image_feature

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io/ioutil"
	"net/http"
	"time"

	"qiniu.com/argus/tuso/search"

	"github.com/pkg/errors"
	"github.com/qiniu/rpc.v1"
	"qiniu.com/argus/tuso/proto"
)

type FeatureApiConfig struct {
	Host          string `json:"host"`
	TimeoutSecond int    `json:"timeout_second"`
}

// TODO: mock it to mock_test.go
type mockImageFeatureAPI struct {
	mode string
}

func (m *mockImageFeatureAPI) PostEvalFeature(ctx context.Context, req proto.PostEvalFeatureReq) (resp *proto.PostEvalFeatureResp, err error) {
	feature := make([]byte, proto.FeatureSize)
	feature[0] = 11
	feature[proto.FeatureSize-1] = 22
	if m.mode == "" {
		return &proto.PostEvalFeatureResp{
			Feature: feature,
			Md5:     "d41d8cd98f00b204e9800998ecf8427e",
		}, nil
	}
	if m.mode == "hasError" {
		if req.Image.Key == "2.jpg" {
			return nil, errors.New("Mock Server Error")
		}
		return &proto.PostEvalFeatureResp{
			Feature: feature,
			Md5:     "d41d8cd98f00b204e9800998ecf8427e",
		}, nil
	}
	panic("bad mode")
}

func NewFeatureApi(cfg FeatureApiConfig) *featureAPi {
	return &featureAPi{
		cfg:    cfg,
		client: http.Client{Timeout: time.Duration(cfg.TimeoutSecond) * time.Second},
	}
}

type featureAPi struct {
	cfg    FeatureApiConfig
	client http.Client
}

func (f *featureAPi) PostEvalFeature(ctx context.Context, req proto.PostEvalFeatureReq) (resp *proto.PostEvalFeatureResp, err error) {
	// curl -v "http://ava-serving-gate.cs.cg.dora-internal.qiniu.io:5001/v1/eval/image-feature" -X "POST" -H "Authorization: QiniuStub uid=1380531519&ut=0" -H "Content-Type: application/json" -d '{"data":{"uri":"qiniu:///vance-test/image.png"}}'
	reqJSON := struct {
		Data struct {
			URI string `json:"uri"`
		} `json:"data"`
	}{}
	reqJSON.Data.URI = fmt.Sprintf("qiniu:///%s/%s", req.Image.Bucket, req.Image.Key)
	if req.Image.Url != "" {
		reqJSON.Data.URI = req.Image.Url
	}
	url := fmt.Sprintf("%s/v1/eval/image-feature", f.cfg.Host)
	buf, _ := json.Marshal(reqJSON)
	httpReq, err := http.NewRequest("POST", url, bytes.NewReader(buf))
	if err != nil {
		return nil, errors.Wrap(err, "PostEvalFeature: http.NewRequest")
	}
	httpReq.Header.Set("Authorization", fmt.Sprintf("QiniuStub uid=%v&ut=0", req.Image.Uid))
	httpReq.Header.Set("Content-Type", "application/json")
	apiResp, err := f.client.Do(httpReq)
	if err != nil {
		return nil, errors.Wrap(err, "PostEvalFeature: post")
	}
	defer apiResp.Body.Close()
	if apiResp.StatusCode/100 == 2 {
		buf, err = ioutil.ReadAll(apiResp.Body)
		if err != nil {
			return nil, errors.Wrap(err, "PostEvalFeature: read body")
		}
		if proto.FeatureSize != len(buf) {
			return nil, errors.New("PostEvalFeature: bad feature size")
		}
		search.NormFeatures(buf, proto.FeatureSize)
		return &proto.PostEvalFeatureResp{Feature: buf}, nil
	}
	return nil, errors.Wrap(rpc.ResponseError(apiResp), "PostEvalFeature")
}
