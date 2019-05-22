package evals

import (
	"context"
	"time"
)

type FaceDetection struct {
	Index int      `json:"index"`
	Class string   `json:"class"`
	Score float32  `json:"score"`
	Pts   [][2]int `json:"pts"`
}

type FaceDetectResp struct {
	Code    int    `json:"code"`
	Message string `json:"message"`
	Result  struct {
		Detections []FaceDetection `json:"detections"`
	} `json:"result"`
}

type FaceDetectReq struct {
	Data struct {
		URI string `json:"uri"`
	} `json:"data"`
	Params struct {
		UseQuality int `json:"use_quality"`
	} `json:"params"`
}

type IFaceDetect interface {
	Eval(context.Context, SimpleReq, uint32, uint32) (FaceDetectResp, error)
}

type _FaceDetect struct {
	_Simple
}

func NewFaceDetect(host string, timeout time.Duration) IFaceDetect {
	return _FaceDetect{_Simple{host: host, path: "/v1/eval/facex-detect", timeout: timeout}}
}

func (e _FaceDetect) Eval(
	ctx context.Context, req SimpleReq, uid, utype uint32,
) (ret FaceDetectResp, err error) {
	err = e._Simple.Eval(ctx, uid, utype, req, &ret)
	return
}

//----------------------------------------------------------------------------//

type FaceReq struct {
	Data struct {
		URI       string `json:"uri"`
		Attribute struct {
			Pts [][2]int `json:"pts"`
		} `json:"attribute,omitempty"`
	} `json:"data"`
}

type IFaceFeature interface {
	Eval(context.Context, FaceReq, uint32, uint32) ([]byte, error)
}

type _FaceFeature struct {
	_SimpleBin
}

func NewFaceFeature(host string, timeout time.Duration, version string) IFaceFeature {
	return _FaceFeature{_SimpleBin{host: host, path: "/v1/eval/facex-feature" + version, timeout: timeout}}
}

func (e _FaceFeature) Eval(
	ctx context.Context, req FaceReq, uid, utype uint32,
) (bs []byte, err error) {
	bs, err = e._SimpleBin.Eval(ctx, uid, utype, req)
	return
}

//----------------------------------------------------------------------------//

type FaceSearchResp struct {
	Code    int    `json:"code"`
	Message string `json:"message"`
	Result  struct {
		Index  int     `json:"index"`
		Class  string  `json:"class"`
		Group  string  `json:"group"`
		Score  float32 `json:"score"`
		Sample struct {
			URL string   `json:"url"`
			Pts [][2]int `json:"pts"`
			ID  string   `json:"id"`
		} `json:"sample"`
	} `json:"result"`
}

type FaceSearchRespV2 struct {
	Code    int    `json:"code"`
	Message string `json:"message"`
	Result  struct {
		Confidences []struct {
			Index  int     `json:"index"`
			Class  string  `json:"class"`
			Group  string  `json:"group"`
			Score  float32 `json:"score"`
			Sample struct {
				URL string   `json:"url"`
				Pts [][2]int `json:"pts"`
				ID  string   `json:"id"`
			} `json:"sample"`
		} `json:"confidences"`
	} `json:"result"`
}

//----------------------------------------------------------------------------//
