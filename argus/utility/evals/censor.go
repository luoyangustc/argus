package evals

import (
	"context"
	"encoding/json"
	"time"
)

type PulpReq struct {
	Data struct {
		URI string `json:"uri"`
	} `json:"data"`
	Params struct {
		Limit int `json:"limit"`
	} `json:"params,omitempty"`
}

type PulpResp struct {
	Code    int    `json:"code"`
	Message string `json:"message"`
	Result  struct {
		Checkpoint  string
		Confidences []struct {
			Index int     `json:"index"`
			Class string  `json:"class"`
			Score float32 `json:"score"`
		} `json:"confidences"`
	} `json:"result"`
}

type IPulp interface {
	Eval(context.Context, PulpReq, uint32, uint32) (PulpResp, error)
}

type _Pulp struct {
	_Simple
}

func (e _Pulp) Eval(ctx context.Context, req PulpReq, uid, utype uint32) (ret PulpResp, err error) {
	err = e._Simple.Eval(ctx, uid, utype, req, &ret)
	return
}

func NewPulp(host string, timeout time.Duration) IPulp {
	return _Pulp{_Simple{host: host, path: "/v1/eval/pulp", timeout: timeout}}
}

//----------------------------------------------------------------------------//
type PulpDetectReq struct {
	Data struct {
		URI string `json:"uri"`
	} `json:"data"`
}

type PulpDetectResp struct {
	Code    int    `json:"code"`
	Message string `json:"message"`
	Result  struct {
		Detections []struct {
			Index int      `json:"index"`
			Class string   `json:"class"`
			Score float32  `json:"score"`
			Pts   [][2]int `json:"pts"`
		} `json:"detections"`
	} `json:"result"`
}

type IPulpDetect interface {
	Eval(context.Context, PulpDetectReq, uint32, uint32) (PulpDetectResp, error)
}

type _PulpDetect struct {
	_Simple
}

func (e _PulpDetect) Eval(ctx context.Context, req PulpDetectReq, uid, utype uint32) (ret PulpDetectResp, err error) {
	err = e._Simple.Eval(ctx, uid, utype, req, &ret)
	return
}

func NewPulpDetect(host string, timeout time.Duration) IPulpDetect {
	return _PulpDetect{_Simple{host: host, path: "/v1/eval/pulp-detect", timeout: timeout}}
}

//----------------------------------------------------------------------------//

type TerrorClassifyResp struct {
	Code    int    `json:"code"`
	Message string `json:"message"`
	Result  struct {
		Confidences []struct {
			Index int     `json:"index"`
			Class string  `json:"class"`
			Score float32 `json:"score"`
		} `json:"confidences"`
	} `json:"result"`
}

type ITerrorClassify interface {
	Eval(context.Context, SimpleReq, uint32, uint32) (TerrorClassifyResp, error)
}

type _TerrorClassify struct {
	_Simple
}

func NewTerrorClassify(host string, timeout time.Duration) ITerrorClassify {
	return _TerrorClassify{_Simple{host: host, path: "/v1/eval/terror-classify", timeout: timeout}}
}

func (e _TerrorClassify) Eval(
	ctx context.Context, req SimpleReq, uid, utype uint32,
) (ret TerrorClassifyResp, err error) {
	err = e._Simple.Eval(ctx, uid, utype, req, &ret)
	return
}

//----------------------------------------------------------------------------//

type TerrorDetection struct {
	Index int      `json:"index"`
	Class string   `json:"class"`
	Score float32  `json:"score"`
	Pts   [][2]int `json:"pts"`
}

type TerrorDetectResp struct {
	Code    int    `json:"code"`
	Message string `json:"message"`
	Result  struct {
		Detections []TerrorDetection `json:"detections"`
	} `json:"result"`
}

type ITerrorDetect interface {
	Eval(context.Context, SimpleReq, uint32, uint32) (TerrorDetectResp, error)
}

type _TerrorDetect struct {
	_Simple
}

func NewTerrorDetect(host string, timeout time.Duration) ITerrorDetect {
	return _TerrorDetect{_Simple{host: host, path: "/v1/eval/terror-detect", timeout: timeout}}
}

func (e _TerrorDetect) Eval(
	ctx context.Context, req SimpleReq, uid, utype uint32,
) (ret TerrorDetectResp, err error) {
	err = e._Simple.Eval(ctx, uid, utype, req, &ret)
	return
}

//----------------------------------------------------------------------------//

type TerrorPreDetection struct {
	Index int      `json:"index"`
	Class string   `json:"class"`
	Score float32  `json:"score"`
	Pts   [][2]int `json:"pts"`
}

type TerrorPreDetectResp struct {
	Code    int    `json:"code"`
	Message string `json:"message"`
	Result  struct {
		Detections []TerrorPreDetection `json:"detections"`
	} `json:"result"`
}

type ITerrorPreDetect interface {
	Eval(context.Context, SimpleReq, uint32, uint32) (TerrorPreDetectResp, error)
}

type _TerrorPreDetect struct {
	_Simple
}

func NewTerrorPreDetect(host string, timeout time.Duration) ITerrorPreDetect {
	return _TerrorPreDetect{_Simple{host: host, path: "/v1/eval/terror-predetect", timeout: timeout}}
}

func (e _TerrorPreDetect) Eval(
	ctx context.Context, req SimpleReq, uid, utype uint32,
) (ret TerrorPreDetectResp, err error) {
	err = e._Simple.Eval(ctx, uid, utype, req, &ret)
	return
}

//----------------------------------------------------------------------------//

type TerrorPostDetection struct {
	Index int      `json:"index"`
	Class string   `json:"class"`
	Score float32  `json:"score"`
	Pts   [][2]int `json:"pts"`
}

type TerrorPostDetectResp struct {
	Code    int    `json:"code"`
	Message string `json:"message"`
	Result  struct {
		Detections []TerrorPostDetection `json:"detections"`
	} `json:"result"`
}

type ITerrorPostDetect interface {
	Eval(context.Context, SimpleReq, uint32, uint32) (TerrorPostDetectResp, error)
}

type _TerrorPostDetect struct {
	_Simple
}

func NewTerrorPostDetect(host string, timeout time.Duration) ITerrorPostDetect {
	return _TerrorPostDetect{_Simple{host: host, path: "/v1/eval/terror-postdet", timeout: timeout}}
}

func (e _TerrorPostDetect) Eval(
	ctx context.Context, req SimpleReq, uid, utype uint32,
) (ret TerrorPostDetectResp, err error) {
	err = e._Simple.Eval(ctx, uid, utype, req, &ret)
	return
}

//----------------------------------------------------------------------------//

type IPolitician interface {
	Eval(context.Context, interface{}, uint32, uint32) (interface{}, error)
}

type _Politician struct {
	_Simple
}

func NewPolitician(host string, timeout time.Duration, suffix string) IPolitician {
	return _Politician{_Simple{host: host, path: "/v1/eval/politician" + suffix, timeout: timeout}}
}

func (e _Politician) Eval(
	ctx context.Context, req interface{}, uid, utype uint32,
) (ret interface{}, err error) {

	var (
		fresp   FaceSearchResp
		frespv2 FaceSearchRespV2
		bs      json.RawMessage
	)

	err = e._Simple.Eval(ctx, uid, utype, req, &bs)
	if err != nil {
		return
	}
	_err := json.Unmarshal(bs, &frespv2)
	if _err == nil && len(frespv2.Result.Confidences) != 0 {
		return frespv2, nil
	}

	err = json.Unmarshal(bs, &fresp)
	if err == nil {
		return fresp, nil
	}
	return
}

//----------------------------------------------------------------------------//
