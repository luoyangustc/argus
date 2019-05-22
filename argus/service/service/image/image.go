// 是否可以移至中间件？
package image

import (
	"bytes"
	"context"
	"encoding/base64"
	"fmt"
	"image"
	"image/jpeg"
	"io"
	"io/ioutil"

	// GIF/JPEG/PNG
	_ "image/gif"
	_ "image/png"

	// BMP
	_ "golang.org/x/image/bmp"

	// WEBP
	_ "golang.org/x/image/webp"

	"github.com/h2non/filetype"
	"qiniu.com/argus/argus/com/uri"
	URI "qiniu.com/argus/com/uri"
	. "qiniu.com/argus/service/service"
)

const (
	MaxSize   = 1024 * 1024 * 10
	MaxWidth  = 4999
	MaxHeight = 4999
)

type STRING string

func (s STRING) String() string {
	if len(s) > 256 {
		return string(s)[:253] + "..."
	}
	return string(s)
}

func (s STRING) GoString() string {
	return s.String()
}

func DataURI(body []byte) STRING {
	return STRING(uri.DataURIPrefix + base64.StdEncoding.EncodeToString(body))
}

type Image struct {
	Format string `json:"format"`
	Width  int    `json:"width"`
	Height int    `json:"height"`
	// Body   []byte  `json:"-"` // 进程内传递统一使用buf
	URI STRING `json:"uri"`
}

type IImageParse interface {
	ParseImage(context.Context, string) (Image, error)
}

type ImageParser struct {
	URI.Handler
}

func NewImageParser() ImageParser {
	return ImageParser{
		Handler: URI.New(
			URI.WithHTTPHandler(),
			URI.WithFileHandler(), // 支持本地文件读取
			URI.WithDataHandler(), // 支持data:application/octet-stream;base64,
		),
	}
}

func (p ImageParser) FetchImage(ctx context.Context, uri string) ([]byte, error) {
	resp, err := p.Get(ctx, URI.Request{URI: uri})
	if err != nil {
		if err == URI.ErrNotSupported {
			return nil, ErrUriNotSupported(fmt.Sprintf("uri not supported: %v", uri))
		} else if err == URI.ErrBadUri {
			return nil, ErrUriBad(fmt.Sprintf("bad uri: %v", uri))
		}
		return nil, ErrUriFetchFailed(fmt.Sprintf("fetch uri failed: %v", uri))
	}
	defer resp.Body.Close()

	if resp.Size > MaxSize {
		return nil, ErrImageTooLarge("image is too large, should be less than 10MB")
	}
	var size int64 = MaxSize + 1
	if resp.Size > 0 {
		size = resp.Size
	}
	bs, err := ioutil.ReadAll(io.LimitReader(resp.Body, size))
	if err != nil {
		return nil, ErrUriFetchFailed(err.Error())
	}
	if len(bs) > MaxSize {
		return nil, ErrImageTooLarge("image is too large, should be less than 10MB")
	}

	return bs, nil
}

func (p ImageParser) ParseImage(ctx context.Context, uri string) (img Image, err error) {

	// var xl = xlog.FromContextSafe(ctx)
	bs, err := p.FetchImage(ctx, uri)
	if err != nil {
		return img, err
	}

	if !filetype.IsImage(bs) {
		// 非图片文件不处理
		return img, ErrImgType("not image")
	}

	// 尝试获取图片大小
	_config, _, err := image.DecodeConfig(bytes.NewReader(bs))
	if err == nil {
		img.Width = _config.Width
		img.Height = _config.Height

		if img.Width > MaxWidth || img.Height > MaxHeight {
			return img, ErrImageTooLarge("image is too large, should be in 4999x4999")
		}
	}

	kind, _ := filetype.Match(bs)
	img.Format = kind.Extension
	if img.Format == "gif" || img.Format == "webp" {
		// 对于gif和webp图片，尝试转成jpeg
		_image, _, err := image.Decode(bytes.NewReader(bs))
		if err == nil {
			var buf = bytes.NewBuffer(nil)
			err = jpeg.Encode(buf, _image, nil)
			if err != nil {
				return img, ErrImgType(err.Error())
			}
			bs = buf.Bytes()
		}
	}
	img.URI = STRING("data:application/octet-stream;base64," + base64.StdEncoding.EncodeToString(bs))

	return img, nil
}
