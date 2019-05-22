package job

import (
	"context"
	"fmt"
	"time"

	"qiniu.com/argus/censor_private/proto"
	"qiniu.com/argus/censor_private/util"
)

// CensorImageService用于请求图片censor服务，并返回结果&错误
type CensorImageService struct {
	host    string
	timeout time.Duration
}

func NewCensorImageService(config *proto.OuterServiceConfig) *CensorImageService {
	return &CensorImageService{
		host:    config.Host,
		timeout: config.Timeout * time.Second,
	}
}

func (s *CensorImageService) Censor(ctx context.Context, url string, scenes []proto.Scene) (
	*proto.ImageCensorResult, *util.ErrorInfo) {
	var req proto.ImageCensorReq
	req.Data.URI = url
	req.Params.Scenes = scenes

	resp := &proto.ImageCensorResp{}
	path := fmt.Sprintf("%s/v3/censor/image", s.host)
	err := util.PostJsonWithCensorError(ctx, s.timeout, path, req, resp)
	if err == nil {
		return &resp.Result, nil
	}

	errInfo, ok := err.(*util.ErrorInfo)
	if ok {
		return nil, errInfo
	}

	return nil, &util.ErrorInfo{Message: err.Error()}
}

type CensorVideoService struct {
	host    string
	timeout time.Duration
}

func NewCensorVideoService(config *proto.OuterServiceConfig) *CensorVideoService {
	return &CensorVideoService{
		host:    config.Host,
		timeout: config.Timeout * time.Second,
	}
}

func (s *CensorVideoService) Censor(ctx context.Context, url string, intervalMsecs int, scenes []proto.Scene) (
	*proto.VideoCensorResult, *util.ErrorInfo) {
	var req proto.VideoCensorReq
	req.Data.URI = url
	req.Params.Scenes = scenes
	req.Params.CutParam.IntervalMsecs = intervalMsecs
	req.Params.Saver.Save = true

	resp := &proto.VideoCensorResp{}
	path := fmt.Sprintf("%s/v3/censor/video", s.host)
	err := util.PostJsonWithCensorError(ctx, s.timeout, path, req, resp)
	if err == nil {
		return &resp.Result, nil
	}

	errInfo, ok := err.(*util.ErrorInfo)
	if ok {
		return nil, errInfo
	}

	return nil, &util.ErrorInfo{Message: err.Error()}
}
