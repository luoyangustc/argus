package gate

import (
	"context"
	"strings"

	xlog "github.com/qiniu/xlog.v1"
	"qiniu.com/argus/com/mime"
	"qiniu.com/argus/com/uri"
	. "qiniu.com/argus/service/service"
)

type GateConfig struct {
	ImageConfig struct {
		Client ClientConfig `json:"client"`
	} `json:"image_config"`
	VideoConfig struct {
		Client   ClientConfig `json:"client"`
		Interval float64      `json:"interval"`
	} `json:"video_config"`
}

type Gate interface {
	Call(context.Context, JsonRequest) (interface{}, error)
}

type gate struct {
	GateConfig
	uri.Handler
	ImageClient Client
	VideoClient Client
}

func NewGate(conf GateConfig) Gate {
	if conf.VideoConfig.Interval == 0 {
		conf.VideoConfig.Interval = 1.0
	}
	return &gate{
		GateConfig:  conf,
		Handler:     uri.New(uri.WithFileHandler(), uri.WithHTTPHandler(), uri.WithDataHandler()),
		ImageClient: NewImageClient(conf.ImageConfig.Client),
		VideoClient: NewVideoClient(conf.VideoConfig.Client, conf.VideoConfig.Interval),
	}
}

func (g *gate) getMime(ctx context.Context, u string) (string, error) {
	xl := xlog.FromContextSafe(ctx)
	resp, err := g.Handler.Get(ctx, uri.Request{URI: u})
	if err != nil {
		xl.Warnf("fail to get uri error: %s", err)
		return "", err
	}

	return mime.ReadMimeType(&resp.Body)
}

func (g *gate) Call(ctx context.Context, req JsonRequest) (interface{}, error) {
	var (
		xl   = xlog.FromContextSafe(ctx)
		resp interface{}
		err  error
	)
	mime, err := g.getMime(ctx, req.Data.URI)
	if err != nil {
		xl.Warnf("parse mime type error: %s", err)
		return nil, ErrUriFetchFailed(err.Error())
	}
	switch {
	case strings.HasPrefix(mime, MIME_IMAGE):
		resp, err = g.ImageClient.CallWithJson(ctx, req)
		return resp, err
	case strings.HasPrefix(mime, MIME_VIDEO):
		resp, err = g.VideoClient.CallWithJson(ctx, req)
		return resp, err
	default:
		xl.Warnf("unsupport mime type: %s", mime)
		return nil, ErrImgType("unsupport mime type " + mime)
	}
}
