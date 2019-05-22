package bucket_inspect

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"strconv"

	"github.com/qiniu/http/httputil.v1"
	xlog "github.com/qiniu/xlog.v1"
	"qiniu.com/argus/censor/biz"
)

func qpulp(ctx context.Context,
	args []string,
	req *struct {
		ReqBody io.ReadCloser
		Cmd     string `json:"cmd"`
		URL     string `json:"url"`
	},
) (interface{}, bool, error) {

	xl := xlog.FromContextSafe(ctx)

	if len(args) == 0 {
		xl.Warnf("bad args: %#v", args)
		return nil, false, httputil.NewError(http.StatusBadRequest, "version info is needed")
	}

	var disable bool

	bs, _ := reqBody(req.URL, req.ReqBody)
	var resp = struct {
		Code    int    `json:"code"`
		Message string `json:"message"`
		Result  struct {
			Label  int     `json:"label"`
			Score  float32 `json:"score"`
			Review bool    `json:"review"`
		} `json:"result"`
	}{}
	xl.Infof("result: %s", string(bs))
	_ = json.Unmarshal(bs, &resp)

	switch args[0] {
	case "v1":

		const (
			KEYPthreshold = "pthreshold"
			KEYSthreshold = "sthreshold"
		)
		var (
			pThreshold float32 // 0.49
			sThreshold float32 = 1.0
		)
		for i := 1; i+1 < len(args); i = i + 2 {
			if v, err := strconv.ParseFloat(args[i+1], 32); err == nil {
				switch args[i] {
				case KEYPthreshold:
					pThreshold = float32(v)
				case KEYSthreshold:
					sThreshold = float32(v)
				}
			} else {
				xlog.Errorf("", "Pulpd parsCmd ParseFloat error:%v", err)
			}
		}

		switch resp.Result.Label {
		case 0:
			disable = resp.Result.Score > 0 && resp.Result.Score >= pThreshold
		case 1:
			disable = resp.Result.Score > 0 && resp.Result.Score >= sThreshold
		}
	default:
		xl.Warnf("bad version: %s", args[0])
		return nil, false, httputil.NewError(http.StatusBadRequest, "bad version")
	}

	return resp, disable, nil

}

////////////////////////////////////////////////////////////////////////////////

func imageCensor(ctx context.Context,
	args []string,
	req *struct {
		ReqBody io.ReadCloser
		Cmd     string `json:"cmd"`
		URL     string `json:"url"`
	},
) (interface{}, bool, error) {

	xl := xlog.FromContextSafe(ctx)

	if len(args) == 0 {
		xl.Warnf("bad args: %#v", args)
		return nil, false, httputil.NewError(http.StatusBadRequest, "version info is needed")
	}

	var disable bool
	var resp interface{}

	bs, _ := reqBody(req.URL, req.ReqBody)
	xl.Infof("result: %s", string(bs))
	xl.Infof("args: %v", args)

	switch args[0] {
	case "v1":

		var respv1 = struct {
			Code    int    `json:"code"`
			Message string `json:"message"`
			Result  struct {
				Label   int     `json:"label"`
				Score   float32 `json:"score"`
				Review  bool    `json:"review"`
				Details []struct {
					Type   string      `json:"type"`
					Label  int         `json:"label"`
					Score  float32     `json:"score"`
					Review bool        `json:"review"`
					More   interface{} `json:"more,omitempty"`
				} `json:"details"`
			} `json:"result"`
		}{}
		_ = json.Unmarshal(bs, &respv1)
		resp = respv1

		const (
			KEY_THRESHOLD_PULP       = "tpulp"
			KEY_THRESHOLD_SEXY       = "tsexy"
			KEY_THRESHOLD_TERROR     = "tterror"
			KEY_THRESHOLD_POLITICIAN = "tpolitician"
		)
		var (
			tp  float32 = 0.6   // 0.6
			ts  float32 = 1.001 // 1.001
			tt  float32 = 0.6   // 0.6
			tpz float32 = 0.6   // 0.6
		)
		for i := 1; i+1 < len(args); i = i + 2 {
			if v, err := strconv.ParseFloat(args[i+1], 32); err == nil {
				switch args[i] {
				case KEY_THRESHOLD_PULP:
					tp = float32(v)
				case KEY_THRESHOLD_SEXY:
					ts = float32(v)
				case KEY_THRESHOLD_TERROR:
					tt = float32(v)
				case KEY_THRESHOLD_POLITICIAN:
					tpz = float32(v)
				}
			} else {
				xlog.Errorf("", "Pulpd parsCmd ParseFloat error:%v", err)
			}
		}

		xl.Infof("Float Parsed: %v, %v, %v, %v", tp, ts, tt, tpz)

		for _, detail := range respv1.Result.Details {
			switch detail.Type {
			case "pulp":
				switch detail.Label {
				case 0:
					disable = disable || (detail.Score > 0 && detail.Score >= tp)
				case 1:
					disable = disable || (detail.Score > 0 && detail.Score >= ts)
				}
			case "terror":
				switch detail.Label {
				case 1:
					disable = disable || (detail.Score > 0 && detail.Score >= tt)
				}
			case "politician":
				switch detail.Label {
				case 1:
					disable = disable || (detail.Score > 0 && detail.Score >= tpz)
				}
			}

			xl.Infof("disable = %v, type = %v, label = %v, score = %v",
				disable, detail.Type, detail.Label, detail.Score)
		}

	case "v2":

		var respv2 = biz.CensorResponse{}
		_ = json.Unmarshal(bs, &respv2)
		resp = respv2

		if len(args) > 1 {
			isDis, _ := strconv.ParseBool(args[1])
			if isDis {
				disable = (respv2.Suggestion == biz.BLOCK)
			}
		}

	default:
		xl.Warnf("bad version: %s", args[0])
		return nil, false, httputil.NewError(http.StatusBadRequest, "bad version")
	}

	return resp, disable, nil
}

////////////////////////////////////////////////////////////////

func videoCensor(ctx context.Context,
	args []string,
	req *struct {
		ReqBody io.ReadCloser
		Cmd     string `json:"cmd"`
		URL     string `json:"url"`
	},
) (interface{}, bool, error) {

	xl := xlog.FromContextSafe(ctx)

	if len(args) == 0 {
		xl.Warnf("bad args: %#v", args)
		return nil, false, httputil.NewError(http.StatusBadRequest, "version info is needed")
	}

	var disable bool
	bs, _ := reqBody(req.URL, req.ReqBody)
	var resp = biz.CensorResponse{}

	xl.Infof("result: %s", string(bs))
	_ = json.Unmarshal(bs, &resp)

	xl.Infof("args: %v", args)
	switch args[0] {
	case "v2":
		if len(args) > 1 {
			isDis, _ := strconv.ParseBool(args[1])
			if isDis {
				disable = (resp.Suggestion == biz.BLOCK)
			}
		}
	default:
		xl.Warnf("bad version: %s", args[0])
		return nil, false, httputil.NewError(http.StatusBadRequest, "bad version")
	}

	return resp, disable, nil
}
