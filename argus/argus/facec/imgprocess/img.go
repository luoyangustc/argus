package imgprocess

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"io/ioutil"
	"net/http"
	"strconv"
	"strings"

	"github.com/pkg/errors"
)

type image struct {
	url string

	newUrl   string
	fileSize int64
	width    int64
	heigth   int64

	infoFetched bool // 图片信息是否被获取过
	err         error
}

//----------------------------------------------------------------------------//

var _ Image = NewThumbnailImage("")

type thumbnailImage struct {
	image
}

func NewThumbnailImage(url string) *thumbnailImage {
	return &thumbnailImage{
		image: image{
			url: url,
		},
	}
}

func (img *thumbnailImage) Fetch(ctx context.Context, client *http.Client) {
	err := img.fetch(ctx, client)
	if err != nil {
		img.err = err
	}
}

func (img *thumbnailImage) fetch(ctx context.Context, client *http.Client) error {
	resp, err := client.Get(addFopCmd(img.url, "imageInfo"))
	if err != nil {
		return errors.Wrap(err, "imageInfo error")
	}
	defer resp.Body.Close()
	if resp.Header.Get("Content-Type") != "application/json" {
		io.Copy(ioutil.Discard, resp.Body)
		return errors.New("imageInfo content type error, only support qiniu bucket url")
	}
	if resp.StatusCode/100 != 2 {
		return errors.New("imageInfo return bad http status")
	}
	type imageInfo struct {
		ColorModel string `json:"colorModel"`
		Format     string `json:"format"`
		Height     int64  `json:"height"`
		Width      int64  `json:"width"`
	}
	var info imageInfo
	err = json.NewDecoder(resp.Body).Decode(&info)
	if err != nil {
		return errors.New("imageInfo result decode error")
	}
	resp, err = client.Head(img.url)
	if err != nil {
		return errors.Wrap(err, "http head origin file error")
	}
	resp, err = client.Head(img.NewUrl())
	if err != nil {
		return errors.Wrap(err, "http head thumbnail image error")
	}
	img.infoFetched = true
	img.fileSize, _ = strconv.ParseInt(resp.Header.Get("Content-Length"), 10, 64)
	img.heigth = info.Height
	img.width = info.Width
	return nil
}

func (img *thumbnailImage) OK() (msg string, ok bool) { return isImageOK(img.image) }
func (img *thumbnailImage) Url() string               { return img.url }
func (img *thumbnailImage) NewUrl() (url string) {
	return addFopCmd(img.url, "imageMogr2/auto-orient/thumbnail/!600x600r")
}

func (img *thumbnailImage) Zoom(pts [][]int64) ([][]int64, [][]int64) {
	l, w := img.width, img.heigth
	if l > w {
		l, w = w, l
	}
	// l < w
	zoom := 600 / float64(l)
	return nil, revPts(pts, zoom)
}

func (img *thumbnailImage) RevertPts(pts [][]int64) [][]int64 {
	l, w := img.width, img.heigth
	if l > w {
		l, w = w, l
	}
	// l < w
	zoom := float64(l) / 600.
	return revPts(pts, zoom)
}

//----------------------------------------------------------------------------//

var _ Image = NewTrimrectImage("", nil)

type trimrectImage struct {
	image
	width  int64
	heigth int64
	pts    [][]int64
}

func NewTrimrectImage(url string, pts [][]int64) *trimrectImage {
	return &trimrectImage{
		image: image{
			url: url,
		},
		pts: pts,
	}
}

func (img *trimrectImage) Fetch(ctx context.Context, client *http.Client) {
	err := img.fetch(ctx, client)
	if err != nil {
		img.err = err
	}
}

func (img *trimrectImage) fetch(ctx context.Context, client *http.Client) error {
	resp, err := client.Get(addFopCmd(img.url, "imageInfo"))
	if err != nil {
		return errors.Wrap(err, "imageInfo error")
	}
	defer resp.Body.Close()
	if resp.Header.Get("Content-Type") != "application/json" {
		io.Copy(ioutil.Discard, resp.Body)
		return errors.New("imageInfo content type error, only support qiniu bucket url")
	}
	if resp.StatusCode/100 != 2 {
		return errors.New("imageInfo return bad http status")
	}
	type imageInfo struct {
		ColorModel string `json:"colorModel"`
		Format     string `json:"format"`
		Height     int64  `json:"height"`
		Width      int64  `json:"width"`
	}
	var info imageInfo
	err = json.NewDecoder(resp.Body).Decode(&info)
	if err != nil {
		return errors.New("imageInfo result decode error")
	}
	resp, err = client.Head(img.NewUrl())
	if err != nil {
		return errors.Wrap(err, "http head thumbnail image error")
	}
	img.infoFetched = true
	img.fileSize, _ = strconv.ParseInt(resp.Header.Get("Content-Length"), 10, 64)
	img.heigth = info.Height
	img.width = info.Width
	return nil
}

func (img *trimrectImage) OK() (msg string, ok bool) { return isImageOK(img.image) }
func (img *trimrectImage) Url() string               { return img.url }
func (img *trimrectImage) NewUrl() (url string) {
	ox, oy := img.pts[0][0], img.pts[0][1]
	w, l := img.pts[1][0]-img.pts[0][0], img.pts[2][1]-img.pts[1][1]
	//if w <= 0 || l <= 0 {
	//	return ""
	//}

	img.image.width = w
	img.image.heigth = l
	return addFopCmd(img.url, fmt.Sprintf("imageMogr2/auto-orient/crop/!%dx%da%da%d", w, l, ox, oy))
}

func (img *trimrectImage) Zoom(pts [][]int64) ([][]int64, [][]int64) {

	var (
		w, l = img.width, img.heigth

		x0, y0 = pts[0][0], pts[0][1]
		x2, y2 = pts[2][0], pts[2][1]
		xo, yo = (x0 + x2) / 2, (y0 + y2) / 2

		npts = make([][]int64, 4)
		opts = make([][]int64, 4)

		nx0, ox0, ny0, oy0, nx2, ox2, ny2, oy2 int64
	)

	if (2*x0 - xo) < 0 {
		nx0, ox0 = 0, x0
	} else {
		nx0, ox0 = 2*x0-xo, xo-x0
	}
	if (2*y0 - yo) < 0 {
		ny0, oy0 = 0, y0
	} else {
		ny0, oy0 = 2*y0-yo, yo-y0
	}
	if (2*x2 - xo) > w {
		nx2, ox2 = w, x2-nx0
	} else {
		nx2, ox2 = 2*x2-xo, x2-nx0
	}
	if (2*y2 - yo) > l {
		ny2, oy2 = l, y2-ny0
	} else {
		ny2, oy2 = 2*y2-yo, y2-ny0
	}

	npts[0] = []int64{nx0, ny0}
	npts[1] = []int64{nx2, ny0}
	npts[2] = []int64{nx2, ny2}
	npts[3] = []int64{nx0, ny2}

	opts[0] = []int64{ox0, oy0}
	opts[1] = []int64{ox2, oy0}
	opts[2] = []int64{ox2, oy2}
	opts[3] = []int64{ox0, oy2}
	return npts, opts
}

func (img *trimrectImage) RevertPts(pts [][]int64) [][]int64 {
	return nil
}

//----------------------------------------------------------------------------//

// 这张图片能否发送给推理API
func isImageOK(img image) (msg string, ok bool) {
	if img.err != nil {
		return img.err.Error(), false
	}
	if img.infoFetched {
		if img.fileSize > 16*1024*1024 {
			return "image to large, max support file size is 16MB", false
		}
		l, w := img.width, img.heigth
		if l > w {
			l, w = w, l
		}
		// l < w
		if l == 0 || w == 0 || l > 10000 || w > 10000 {
			return "image resolution ratio error", false
		}
		if w/l > 5 {
			return "image resolution ratio error", false
		}
	}
	return "", true
}

// addFopCmd 在URL后面拼接dora的fop命令
func addFopCmd(originUrl string, cmd string) string {
	if strings.HasSuffix(originUrl, "?") {
		return originUrl + cmd
	}
	if strings.Contains(originUrl, "?") {
		return originUrl + "|" + cmd
	}
	return originUrl + "?" + cmd
}

func revPts(pts [][]int64, zoom float64) (r [][]int64) {
	for _, i := range pts {
		l := make([]int64, len(i))
		for i, j := range i {
			l[i] = int64(float64(j) * zoom)
		}
		r = append(r, l)
	}
	return
}
