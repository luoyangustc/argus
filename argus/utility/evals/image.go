package evals

import (
	"context"
	"time"
)

type ImageInfoResp struct {
	Code    int    `json:"code"`
	Message string `json:"message"`
	Result  struct {
		Metadata struct {
			Format string `json:"format"`
			Width  int    `json:"width"`
			Height int    `json:"height"`
		} `json:"metadata"`
	} `json:"result"`
}

type IImageInfo interface {
	Eval(context.Context, SimpleReq, uint32, uint32) (ImageInfoResp, error)
}

type _ImageInfo struct {
	_Simple
}

func (e _ImageInfo) Eval(
	ctx context.Context, req SimpleReq, uid, utype uint32,
) (ret ImageInfoResp, err error) {
	err = e._Simple.Eval(ctx, uid, utype, req, &ret)
	return
}

func NewImageInfo(host string, timeout time.Duration) IImageInfo {
	return _ImageInfo{_Simple{host: host, path: "/v1/eval/image", timeout: timeout}}
}

//-----------------------------------------------------------------------------//

type IImageFeature interface {
	Eval(context.Context, SimpleReq, uint32, uint32) ([]byte, error)
}

type _ImageFeature struct {
	_SimpleBin
}

func NewImageFeature(host string, timeout time.Duration, version string) IImageFeature {
	return _ImageFeature{_SimpleBin{host: host, path: "/v1/eval/image-feature" + version, timeout: timeout}}
}

func (e _ImageFeature) Eval(
	ctx context.Context, req SimpleReq, uid, utype uint32,
) (bs []byte, err error) {
	bs, err = e._SimpleBin.Eval(ctx, uid, utype, req)
	return
}

//-----------------------------------------------------------------------------//
