package server

import (
	"bytes"
	"context"
	"encoding/base64"
	"image"
	"image/jpeg"
	"io"
	"strings"

	// GIF/JPEG/PNG
	_ "image/gif"
	_ "image/png"
	// BMP
	_ "golang.org/x/image/bmp"

	"github.com/qiniu/xlog.v1"

	URI "qiniu.com/argus/com/uri"
	STS "qiniu.com/argus/sts/client"
)

type Image struct {
	Format string  `json:"format"`
	Width  int     `json:"width"`
	Height int     `json:"height"`
	URI    *string `json:"uri,omitempty"`
}

type IImageParse interface {
	ParseImage(context.Context, string) (Image, error)
}

type ImageParse struct {
	STS.Client
}

func (p ImageParse) ParseImage(ctx context.Context, uri string) (img Image, err error) {
	const (
		MaxSize = 1024 * 1024 * 10
	)

	var (
		xl = xlog.FromContextSafe(ctx)
		r  io.Reader
	)

	switch {
	case strings.HasPrefix(uri, _DataURIPrefix):
		bs, err := base64.StdEncoding.DecodeString(strings.TrimPrefix(uri, _DataURIPrefix))
		if err != nil {
			return img, err
		}
		if len(bs) > MaxSize {
			return img, ErrorImageTooLarge
		}
		r = bytes.NewReader(bs)
	default:
		url, _ := p.Client.GetURL(ctx, uri, nil, nil)
		r1, _, _, err := p.Client.Get(ctx, uri, nil)
		if err != nil {
			return img, err
		}
		defer r1.Close()
		var bs = make([]byte, MaxSize+1)
		n, err := io.ReadFull(r1, bs)
		if n == 0 {
			return img, err
		}
		if err != io.ErrUnexpectedEOF {
			if err == nil {
				err = ErrorImageTooLarge
			}
			return img, err
		}
		r = bytes.NewReader(bs[:n])
		img.URI = &url
	}

	_image, format, err := image.Decode(r)
	if err != nil {
		return img, err
	}

	img.Format = format
	img.Width = _image.Bounds().Dx()
	img.Height = _image.Bounds().Dy()

	if img.Format == "gif" {
		var buf = bytes.NewBuffer(nil)
		err = jpeg.Encode(buf, _image, nil)
		if err != nil {
			return img, err
		}
		var n = int64(buf.Len())
		_uri, err := p.Client.NewURL(ctx, &n)
		if err != nil {
			xl.Errorf("new url failed. %v", err)
			return img, err
		}
		if err := p.Client.Post(ctx, _uri, n, buf); err != nil {
			return img, err
		}
		img.URI = &_uri
	}

	return img, nil
}

type StandaloneImageParser struct {
	URI.Handler
}

func NewStandaloneImageParser() StandaloneImageParser {
	return StandaloneImageParser{
		Handler: URI.New(
			URI.WithHTTPHandler(),
			URI.WithFileHandler(), // 支持本地文件读取
			URI.WithDataHandler(), // 支持data:application/octet-stream;base64,
		),
	}
}

func (s StandaloneImageParser) ParseImage(ctx context.Context, uri string) (img Image, err error) {
	resp, err := s.Get(ctx, URI.Request{URI: uri})
	if err != nil {
		return img, err
	}
	defer resp.Body.Close()

	_image, format, err := image.Decode(resp.Body)
	if err != nil {
		return img, err
	}

	img.Format = format
	img.Width = _image.Bounds().Dx()
	img.Height = _image.Bounds().Dy()

	if img.Format == "gif" {
		var buf = bytes.NewBuffer(nil)
		err = jpeg.Encode(buf, _image, nil)
		if err != nil {
			return img, err
		}
		_uri := "data:application/octet-stream;base64," + base64.StdEncoding.EncodeToString(buf.Bytes())
		img.URI = &_uri
	}

	return
}
