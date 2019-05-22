package wangan_mix

import (
	"context"
	"encoding/base64"
	"fmt"
	"io/ioutil"
	"os"
	"path"
	"strconv"
	"strings"
	"time"

	"github.com/go-kit/kit/endpoint"
	xlog "github.com/qiniu/xlog.v1"
	. "qiniu.com/argus/service/service"
	"qiniu.com/argus/service/service/image"
)

// 分类与细分类

type MixClasses map[string][]string

func (m MixClasses) Doc() string {
	str := `**请求type包含的子类别说明：**
| Type选项 | 识别的子类别 |
| :--- | :--- |
`
	for t, classes := range m {
		class_str := strings.Join(classes, ", ")
		str += fmt.Sprintf("| %s | %s |\n", t, class_str)
	}
	str += "| all | 上述所有子类别 |\n"
	return str
}

var (
	MIX_CLASSES MixClasses = map[string][]string{
		"terror": []string{
			"beheaded_isis", "beheaded_decollation", "knives_true", "guns_true", "BK_LOGO_1",
			"BK_LOGO_2", "BK_LOGO_3", "BK_LOGO_4", "BK_LOGO_5", "BK_LOGO_6", "isis_flag",
			"islamic_flag", "tibetan_flag", "falungong_logo",
		},
		"internet_terror": []string{
			"bloodiness_human", "bomb_fire", "bomb_smoke", "bomb_vehicle", "bomb_self-burning",
			"beheaded_isis", "beheaded_decollation", "march_banner", "march_crowed", "fight_police",
			"fight_person", "character", "masked", "army", "scene_person", "islamic_dress",
			"knives_true", "guns_true", "BK_LOGO_1", "BK_LOGO_2", "BK_LOGO_3", "BK_LOGO_4",
			"BK_LOGO_5", "BK_LOGO_6", "isis_flag", "islamic_flag", "tibetan_flag", "falungong_logo",
		},
		"certificate": []string{
			"idcard_positive", "idcard_negative", "bankcard_positive", "bankcard_negative", "gongzhang",
		},
	}
)

type WanganMixReq struct {
	Data struct {
		IMG image.Image
	} `json:"data"`
	Params struct {
		Detail bool   `json:"detail"`
		Type   string `json:"type"`
	} `json:"params"`
}

type WanganMixResp struct {
	Code    int             `json:"code"`
	Message string          `json:"message"`
	Result  WanganMixResult `json:"result"`
}

type WanganMixResult struct {
	Label     int                  `json:"label"`
	Score     float32              `json:"score"`
	Classes   []string             `json:"classes,omitempty"`
	Classify  []WanganMixClassify  `json:"classify,omitempty"`
	Detection []WanganMixDetection `json:"detection,omitempty"`
}

type WanganMixClassify struct {
	Class string  `json:"class"`
	Score float32 `json:"score"`
}

type WanganMixDetection struct {
	Class string    `json:"class"`
	Score float32   `json:"score"`
	Pts   [4][2]int `json:"pts"`
}

type WanganMixService interface {
	WanganMix(context.Context, WanganMixReq) (WanganMixResp, error)
}

var _ WanganMixService = WanganMixEndPoints{}

type WanganMixEndPoints struct {
	WanganMixEP endpoint.Endpoint
}

func (ends WanganMixEndPoints) WanganMix(ctx context.Context, req WanganMixReq) (WanganMixResp, error) {
	resp, err := ends.WanganMixEP(ctx, req)
	if err != nil {
		return WanganMixResp{}, err
	}
	return resp.(WanganMixResp), nil
}

type Config struct {
	Savespace string `json:"savespace"`
}

var (
	DEFAULT = Config{}
)

type wanganmixService struct {
	Config
	EvalWanganMixService
}

func NewWanganMixService(
	conf Config,
	srv EvalWanganMixService,
) (WanganMixService, error) {
	return wanganmixService{
		Config:               conf,
		EvalWanganMixService: srv,
	}, nil
}

func inTypes(class string, types []string) bool {
	if len(types) == 0 {
		return true
	}
	for _, t := range types {
		if class == t {
			return true
		}
	}
	return false
}

func (s wanganmixService) WanganMix(ctx context.Context, req WanganMixReq) (WanganMixResp, error) {

	var (
		xl          = xlog.FromContextSafe(ctx)
		req1        EvalWanganMixReq
		resp1       EvalWanganMixResp
		err         error
		normalScore float32
		classes     = make(map[string]float32, 0)
		ret         WanganMixResp
	)

	req1.Data.IMG.URI = req.Data.IMG.URI
	resp1, err = s.EvalWanganMix(ctx, req1)
	if err != nil {
		xl.Errorf("call /v1/eval/wangan-mix error resp: %v", err)
		err = ErrInternal(err.Error())
		return WanganMixResp{}, err
	}

	types := MIX_CLASSES[req.Params.Type]
	for _, cl := range resp1.Result.Classify.Confidences {
		if cl.Class == "normal" || inTypes(cl.Class, types) {
			ret.Result.Classify = append(ret.Result.Classify, struct {
				Class string  `json:"class"`
				Score float32 `json:"score"`
			}{
				Class: cl.Class,
				Score: cl.Score,
			})
			if cl.Class != "normal" {
				score := classes[cl.Class]
				if cl.Score > score {
					classes[cl.Class] = cl.Score
				}
			} else {
				if cl.Score > normalScore {
					normalScore = cl.Score
				}
			}
		}
	}
	for _, dt := range resp1.Result.Detection {
		if dt.Class != "not_terror" && inTypes(dt.Class, types) {
			ret.Result.Detection = append(ret.Result.Detection, struct {
				Class string    `json:"class"`
				Score float32   `json:"score"`
				Pts   [4][2]int `json:"pts"`
			}{
				Class: dt.Class,
				Score: dt.Score,
				Pts:   dt.Pts,
			})
			score := classes[dt.Class]
			if dt.Score > score {
				classes[dt.Class] = dt.Score
			}
		} else {
			if dt.Score > normalScore {
				normalScore = dt.Score
			}
		}
	}

	if len(classes) > 0 {
		ret.Result.Label = 1
		for class, score := range classes {
			ret.Result.Classes = append(ret.Result.Classes, class)
			if score > ret.Result.Score {
				ret.Result.Score = score
			}
		}
	} else {
		ret.Result.Label = 0
		ret.Result.Score = normalScore
		ret.Result.Classes = append(ret.Result.Classes, "normal")
	}

	if ret.Result.Label == 1 {
		var file string
		if s.Config.Savespace != "" {
			// 保存识别出来的原图
			dir := time.Now().Format("2006010203")
			file = path.Join(dir, strconv.FormatInt(time.Now().UnixNano(), 10))
			os.MkdirAll(path.Join(s.Config.Savespace, dir), 0755)
			buf, e := base64.StdEncoding.DecodeString(strings.TrimPrefix(string(req.Data.IMG.URI), "data:application/octet-stream;base64,"))
			if e != nil {
				xl.Warnf("fail to base64 decode uri, error: %s", e.Error())
				return ret, nil
			}
			if e = ioutil.WriteFile(path.Join(s.Config.Savespace, file), buf, 0755); e != nil {
				xl.Warnf("fail to create image file, error: %s", e.Error())
				return ret, nil
			}
		}
		xl.Infof("save image %s, result %#v", file, ret.Result)
	}

	if !req.Params.Detail {
		ret.Result.Classify = ret.Result.Classify[:0]
		ret.Result.Detection = ret.Result.Detection[:0]
	}

	return ret, nil
}
