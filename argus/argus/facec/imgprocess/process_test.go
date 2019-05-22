package imgprocess

import (
	"context"
	"encoding/json"
	"net/http"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/stretchr/testify.v2/assert"
)

// travis 访问国内太慢， 不跑测试
var skipImgTest = os.Getenv("TRAVIS") != ""

var client = &http.Client{
	Timeout: time.Second * 5,
}

func Test2Image(t *testing.T) {
	if skipImgTest {
		t.SkipNow()
	}
	s := New([]Image{
		NewThumbnailImage("http://oohbr1ly0.bkt.clouddn.com/demo5/99000479601529/9994e7ff22ac2cfd74412289bf3bb1d8.jpg"),
		NewThumbnailImage("http://oohbr1ly0.bkt.clouddn.com/demo5/99000479601529/85c377e893a0b1c62e1374ea1691119d.jpg"),
	}, client)
	s.FetchAll(context.Background())
	assert.Len(t, s.NewUrls(), 2)
	assert.Len(t, s.BadUrls(), 0)
}

func Test5Image(t *testing.T) {
	if skipImgTest {
		t.SkipNow()
	}
	s := New([]Image{
		NewThumbnailImage("http://oohbr1ly0.bkt.clouddn.com/demo5/99000479601529/9994e7ff22ac2cfd74412289bf3bb1d8.jpg"),
		NewThumbnailImage("http://oohbr1ly0.bkt.clouddn.com/demoyo8/867686022386195/c5f1ad563ae532fd1ef51ff9d0359931.png"),
		NewThumbnailImage("http://oohbr1ly0.bkt.clouddn.com/demoyo8/867686022386195/c4b53839600575e523e693211d8604f4.png"),
		NewThumbnailImage("http://oohbr1ly0.bkt.clouddn.com/demo5/99000479601529/85c377e893a0b1c62e1374ea1691119d.jpg"),
		NewThumbnailImage("http://oohbr1ly0.bkt.clouddn.com/demoyo8/867686022386195/ffd38e6d91871e54eb753e3a1fce6d83.png"),
	}, client)
	s.FetchAll(context.Background())
	assert.Len(t, s.NewUrls(), 5)
	assert.Len(t, s.BadUrls(), 0)
}

func Test2BadImg(t *testing.T) {
	if skipImgTest {
		t.SkipNow()
	}
	s := New([]Image{
		NewThumbnailImage("fdsfsdf"),                                                                                  // 非法URL
		NewThumbnailImage("https://ss0.bdstatic.com/5aV1bjqh_Q23odCf/static/superman/img/logo/bd_logo1_31bdc765.png"), // 非七牛图片
		NewThumbnailImage("http://q.hi-hi.cn/dcdsfsdfsdfsfdfs"),                                                       // 不存在
		// NewThumbnailImage("http://q.hi-hi.cn/qiniu-hr.mp4"),                                                           // 非图片
	}, client)
	s.FetchAll(context.Background())
	assert.Len(t, s.NewUrls(), 0)
	assert.Len(t, s.BadUrls(), 3)
	assert.Equal(t, s.BadUrls()[0].URL, "fdsfsdf")
	assert.Equal(t, s.BadUrls()[1].URL, "https://ss0.bdstatic.com/5aV1bjqh_Q23odCf/static/superman/img/logo/bd_logo1_31bdc765.png")
	assert.Contains(t, s.BadUrls()[0].Err, "unsupported protocol")
	assert.Contains(t, s.BadUrls()[1].Err, "only support qiniu bucket")
	assert.Contains(t, s.BadUrls()[2].Err, "bad http status")
	// assert.Contains(t, s.BadUrls()[3].Err, "bad http status")
}

func Test29Img(t *testing.T) {
	if skipImgTest {
		t.SkipNow()
	}
	urls := strings.Split(`https://dn-group-photos.qbox.me/Frsjikj08PeG46cUEvuO_swTFprp
https://dn-group-photos.qbox.me/Fn8wk7hJAh4rKZIbRWjjPo1m-YQa
https://dn-group-photos.qbox.me/FoxbiHjSAqOdm3B3ZljHi1aqbXFT
https://dn-group-photos.qbox.me/FsxIagymTJIhiTbyAwywYIrvk1id
https://dn-group-photos.qbox.me/Fut0G_JPBpYo_vEwNbUjB9WUbWkd
https://dn-group-photos.qbox.me/Fv-j2pLtFrKhYBMNWXM7sZxWSKqb
https://dn-group-photos.qbox.me/FtsHQao52ASS_bAJmOCPwf1_Xjqn
https://dn-group-photos.qbox.me/FqeMrY7ADMe2dD5rB4JDQuYg-NBo
https://dn-group-photos.qbox.me/FlMcMS3QAiXQcxVBkHYXBIGZeH7-
https://dn-group-photos.qbox.me/FskM6jIay1dPdrIBiPY_YQTHDITS
https://dn-group-photos.qbox.me/Fvam7NOpqO0uc4emHZ0kATjh3B3P
https://dn-group-photos.qbox.me/Fn1vqbgEiEBcUMZnVMrjhrEIhQk2
https://dn-group-photos.qbox.me/FuQT-AVm4EpZsN-GwQcyELabwbcO
https://dn-group-photos.qbox.me/FuemEqCfMMKY-3mkkrLAvagv9_Eq
https://dn-group-photos.qbox.me/Fp_H6bGlNlHgNKcrcSkS7rz0N_La
https://dn-group-photos.qbox.me/Fhq82z64qQFtZvdeTbJQhXf7t_jZ
https://dn-group-photos.qbox.me/Ftg77exF5_liD3HNo5RJUzn6Y6Gk
https://dn-group-photos.qbox.me/FtdtDri6VLnOZNLOA6zVRM0Se18h
https://dn-group-photos.qbox.me/FkU1yAEOr3WcueY492INLkZFgEmz
https://dn-group-photos.qbox.me/Fgf13XvetJJOet5fTJnoK1rVRall
https://dn-group-photos.qbox.me/ls-JoG5GeYdHY3_T92vMF-Xg4Nrz
https://dn-group-photos.qbox.me/lrmRK1MRGoE_SSs-46JJKeiyFz2O
https://dn-group-photos.qbox.me/lslWjivvde477u9KW3-OUxhZ_BLr
https://dn-group-photos.qbox.me/ljY-3DJTTY7wlurbP8G4_bay7GW8
https://dn-group-photos.qbox.me/lu829Vl0BOWuuRAX7QHC9roL4EEW
https://dn-group-photos.qbox.me/lrkXrD8CLMXixH9E_ntWRh2Ep0mS
https://dn-group-photos.qbox.me/Fpxy9Ly173ddQiwewcuicr1wTOF9
https://dn-group-photos.qbox.me/FpoNri3JIfuPqseBTZaAaxMwgyn-
https://dn-group-photos.qbox.me/FlaQnYFoSvG7GFSDWs5PhLD6LrYn`, "\n")
	images := make([]Image, 0, len(urls))
	for _, url := range urls {
		images = append(images, NewThumbnailImage(url))
	}
	s := New(images, client)
	s.FetchAll(context.Background())
	assert.Len(t, s.NewUrls(), len(urls))
	assert.Len(t, s.BadUrls(), 0)
}

func Test_revPts(t *testing.T) {
	j := "[[137,131],[411,131],[411,599],[137,599]]"
	var a [][]int64
	json.Unmarshal([]byte(j), &a)
	r := revPts(a, 2)
	assert.EqualValues(t, r[0][0], 274)
}

func Test_addFopCmd(t *testing.T) {
	assert.Equal(t, addFopCmd("https://dn-group-photos.qbox.me/FkhLuHRDHAF5W8cMK5OQ2100R_XU", "vinfo"), "https://dn-group-photos.qbox.me/FkhLuHRDHAF5W8cMK5OQ2100R_XU?vinfo")
	assert.Equal(t, addFopCmd("https://dn-group-photos.qbox.me/FkhLuHRDHAF5W8cMK5OQ2100R_XU?vinfo", "fdsfsdf"), "https://dn-group-photos.qbox.me/FkhLuHRDHAF5W8cMK5OQ2100R_XU?vinfo|fdsfsdf")
	assert.Equal(t, addFopCmd("https://dn-group-photos.qbox.me/FkhLuHRDHAF5W8cMK5OQ2100R_XU?", "fdsfsdf"), "https://dn-group-photos.qbox.me/FkhLuHRDHAF5W8cMK5OQ2100R_XU?fdsfsdf")
}
