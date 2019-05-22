package imgprocess

import (
	"context"
	"net/http"
	"sync"
	"time"

	"github.com/qiniu/xlog.v1"
)

type Image interface {
	Fetch(context.Context, *http.Client)
	OK() (string, bool)
	Url() string
	NewUrl() string
	Zoom(pts [][]int64) ([][]int64, [][]int64)
	RevertPts([][]int64) [][]int64
}

type ImgProcesser interface {
	FetchAll(context.Context)
	NewUrls() []string
	BadUrls() []BadUrl

	Find(string) Image
	Revert(string) Image
}

var _ ImgProcesser = New([]Image{}, nil)

type imgProcess struct {
	images []Image
	client *http.Client
}

func New(images []Image, client *http.Client) ImgProcesser {
	return &imgProcess{
		images: images,
		client: client,
	}
}

func (s *imgProcess) FetchAll(ctx context.Context) {
	xl := xlog.FromContextSafe(ctx)
	defer func(begin time.Time) {
		duration := time.Since(begin)
		xl.Xprof2("IMGP", duration, nil)
		xl.Infof("process images. %d %d", len(s.images), duration/time.Millisecond)
	}(time.Now())

	w := sync.WaitGroup{}
	w.Add(len(s.images))
	for _, v := range s.images {
		go func(v Image) {
			v.Fetch(ctx, s.client)
			w.Done()
		}(v)
	}
	w.Wait()
}

func (s *imgProcess) NewUrls() []string {
	urls := make([]string, 0)
	for _, image := range s.images {
		if _, ok := image.OK(); !ok {
			continue
		}
		url := image.NewUrl()
		urls = append(urls, url)
	}
	return urls
}

type BadUrl struct {
	URL string
	Err string
}

func (s *imgProcess) BadUrls() (b []BadUrl) {
	for _, image := range s.images {
		msg, ok := image.OK()
		if !ok {
			b = append(b, BadUrl{URL: image.Url(), Err: msg})
		}
	}
	return
}

func (s *imgProcess) Find(url string) Image {
	for _, img := range s.images {
		if img.Url() == url {
			return img
		}
	}
	return nil
}

func (s *imgProcess) Revert(url string) Image {
	for _, img := range s.images {
		if img.NewUrl() == url {
			return img
		}
	}
	return nil
}

//----------------------------------------------------------------------------//

var _ ImgProcesser = NewMockProcess([]Image{}, nil)

type MockImgProcess struct {
	images []Image
}

func NewMockProcess(images []Image, client *http.Client) ImgProcesser {
	return &MockImgProcess{images: images}
}

func (s *MockImgProcess) FetchAll(context.Context) {}

func (s *MockImgProcess) NewUrls() []string {
	urls := make([]string, 0, len(s.images))
	for _, image := range s.images {
		urls = append(urls, image.NewUrl())
	}
	return urls
}

func (s *MockImgProcess) BadUrls() []BadUrl { return []BadUrl{} }

func (s *MockImgProcess) Find(url string) Image {
	for _, image := range s.images {
		if image.Url() == url {
			return image
		}
	}
	return nil
}

func (s *MockImgProcess) Revert(url string) Image {
	for _, image := range s.images {
		if image.NewUrl() == url {
			return image
		}
	}
	return NewThumbnailImage(url)
}
