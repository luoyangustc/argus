package service

import (
	"context"
	"path"
	"time"

	"github.com/qiniu/rpc.v3"
)

const (
	ZhatuTypeCovered   = 0
	ZhatuTypeUncovered = 1
	// TODO cover status
)

type Zhatu interface {
	Eval(context.Context, EvalZhatuReq) (EvalZhatuResp, error)
}

type _Zhatu struct {
	url     string
	timeout time.Duration
	*rpc.Client
}

func NewZhatu(conf EvalConfig) _Zhatu {
	url := conf.Host + "/v1"
	if conf.URL != "" {
		url = conf.URL
	}
	return _Zhatu{url: url, timeout: time.Duration(conf.Timeout) * time.Second}
}

func (zt _Zhatu) eval(ctx context.Context, method, uri string, req interface{}, resp interface{}) (err error) {
	var (
		client *rpc.Client
	)
	if zt.Client == nil {
		client = NewClient(zt.timeout)
	} else {
		client = zt.Client
	}
	err = callRetry(ctx,
		func(ctx context.Context) error {
			var err1 error
			err1 = client.CallWithJson(ctx, resp, method, zt.url+uri, req)
			return err1
		})
	return
}

// -----------------------------------------------------------------------------------
type EvalZhatuReq struct {
	Data struct {
		URI       string `json:"uri"`
		Attribute struct {
			Name      string    `json:"name, omitempty"`
			ImageType int       `json:"image_type, omitempty"`
			LanePTS   [4][2]int `json:"lane_pts, omitempty"`
			Video     bool      `json:"video"`
		}
	} `json:"data"`
}

type EvalZhatuDetection struct {
	Label int     `json:"label"`
	Class int     `json:"class,omitempty"`
	Score float32 `json:"score"`
	PTS   [4]int  `json:"pts"`
}

type EvalZhatuResp struct {
	Code    int    `json:"code"`
	Message string `json:"message"`
	Result  struct {
		Detections []EvalZhatuDetection `json:"detections"`
	} `json:"result"`
}

func (zt _Zhatu) Eval(
	ctx context.Context, req EvalZhatuReq,
) (resp EvalZhatuResp, err error) {
	//uri := "/eval/zhatu"
	uri := "/eval"
	err = zt.eval(ctx, "POST", uri, req, &resp)
	return
}

// -----------------------------------------------------------------------------------
type Zhongxing interface {
	Zhatu(context.Context, Capture) error
	Report(ctx context.Context, startTime, endTime string) ([]reportZhatucheCaptureResp, error)
}

type _Zhongxing struct {
	url     string
	timeout time.Duration
	*rpc.Client
}

func NewZhongxing(conf EvalConfig) _Zhongxing {
	url := conf.Host + "/v1"
	if conf.URL != "" {
		url = conf.URL
	}
	return _Zhongxing{url: url, timeout: time.Duration(conf.Timeout) * time.Second}
}

func (zx _Zhongxing) Zhatu(ctx context.Context, req Capture) (err error) {
	var (
		client *rpc.Client
	)
	if zx.Client == nil {
		client = NewClient(zx.timeout)
	} else {
		client = zx.Client
	}

	uri := zx.url + "/zhatuche/capture/" + req.ID
	req.ID = ""
	err = callRetry(ctx,
		func(ctx context.Context) error {
			var err1 error
			err1 = client.CallWithJson(ctx, nil, "POST", uri, req)
			return err1
		})
	return
}

type reportZhatucheCaptureResp struct {
	ID          string  `json:"id"`
	Time        string  `json:"time"`
	CameraID    string  `json:"camera_id"`
	CameraInfo  string  `json:"camera_info"`
	LicenceID   string  `json:"licence_id"`
	LicenceType string  `json:"licence_type"`
	Lane        int     `json:"lane"`
	Result      int     `json:"result"`
	Score       float32 `json:"score"`
	Coordinate  struct {
		GPS [2]float32 `json:"gps"`
	} `json:"coordinate"`
	Resource struct {
		Images []struct {
			URI      string `json:"uri"`
			FileName string `json:"filename"`
			PTS      [4]int `json:"pts"`
		} `json:"images"`
		Videos []string `json:"videos"`
	} `json:"resource"`
}

func (zx _Zhongxing) Report(ctx context.Context, startTime, endTime string) (resp []reportZhatucheCaptureResp, err error) {
	var (
		client *rpc.Client
	)
	if zx.Client == nil {
		client = NewClient(zx.timeout)
	} else {
		client = zx.Client
	}

	uri := zx.url + "/zhatuche/start/" + startTime + "/end/" + endTime
	err = callRetry(ctx,
		func(ctx context.Context) error {
			var err1 error
			err1 = client.CallWithJson(ctx, &resp, "GET", uri, nil)
			return err1
		})
	if err == nil {
		for i, capture := range resp {
			for j, image := range capture.Resource.Images {
				image.FileName = path.Base(image.URI)
				capture.Resource.Images[j] = image
			}
			resp[i] = capture
		}
	}
	return
}

// -----------------------------------------------------------------------------------
type Jiaoguan interface {
	Jiaoguan(context.Context, JiaoguanReq) (JiaoguanResp, error)
}

type _Jiaoguan struct {
	url     string
	timeout time.Duration
	*rpc.Client
}

func NewJiaoguan(conf EvalConfig) _Jiaoguan {
	url := conf.Host + "/v1"
	if conf.URL != "" {
		url = conf.URL
	}
	return _Jiaoguan{url: url, timeout: time.Duration(conf.Timeout) * time.Second}
}

type JiaoguanReq struct {
	StartTime int64  `json:"start_time"`
	CameraID  string `json:"camera_id"`
	CameraIP  string `json:"camera_ip"`
	Duration  int    `json:"duration"`
}

type JGImage struct {
	ID           string  `json:"TPID"`
	Timestamp    float64 `json:"JGSJ"`
	CameraID     string  `json:"DMDM"`
	Attribution  string  `json:"CLSD"`
	TimestampStr string  `json:"JGSJ_STR"`
	Lane         int     `json:"CDBH"`
	LicenceID    string  `json:"HPHM"`
	LicenceType  string  `json:"HPYS"`
	LicenceImage string  `json:"licence_image,omitempty"`
	FullImage    string  `json:"file_name"`
}

type JiaoguanResp struct {
	Images  []JGImage `json:"imgs"`
	BaseURL string    `json:"base_url"`
	Param   struct {
		CameraID  string  `json:"camera_id"`
		CameraIP  string  `json:"camera_ip"`
		Duration  int     `json:"duration"`
		StartTime float32 `json:"start_time"`
	} `json:"param"`
}

func (jg _Jiaoguan) Jiaoguan(ctx context.Context, req JiaoguanReq) (resp JiaoguanResp, err error) {
	var (
		client *rpc.Client
	)
	if jg.Client == nil {
		client = NewClient(jg.timeout)
	} else {
		client = jg.Client
	}

	uri := jg.url + "/fetch_imgs"
	err = callRetry(ctx,
		func(ctx context.Context) error {
			var err1 error
			err1 = client.CallWithJson(ctx, &resp, "POST", uri, req)
			return err1
		})
	return
}
