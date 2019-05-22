package eval

import (
	"context"
	"encoding/json"
	"io"
	"os"
	"path"
	"strings"

	"github.com/pkg/errors"
	"github.com/qiniu/xlog.v1"

	"qiniu.com/argus/atserving/model"
	"qiniu.com/argus/atserving/tensor"
	"qiniu.com/argus/com/uri"
	"qiniu.com/argus/com/util"
)

const (
	TensorTypeEnv  = "TENSOR_TYPE"
	tensorClassify = "classify"
	tensorDetect   = "detect"
)

var (
	ErrInvalidTensorType = errors.New("invalid tensorrt type")
)

var _ Core = &EvalNet{}

// TensorrtEvalNet ...
type TensorrtEvalNet struct {
	cfg        *EvalConfig
	net        interface{}
	tensorType string
}

func (en TensorrtEvalNet) fetchFile(ctx context.Context, url, local string) (err error) {
	xl := xlog.FromContextSafe(ctx)
	if strings.HasPrefix(url, "file://") {
		var src, dest *os.File
		src, err = os.Open(strings.TrimPrefix(url, "file://"))
		if err != nil {
			xl.Warnf("open file failed. %s %v", strings.TrimPrefix(url, "file://"), err)
			return
		}
		defer src.Close()
		dest, err = os.Create(local)
		if err != nil {
			xl.Warnf("create file failed. %s %v", local, err)
			return
		}
		defer dest.Close()
		_, err = io.Copy(dest, src)
		return
	}
	var c uri.Handler
	if len(en.cfg.Fetcher.AK) > 0 {
		c = uri.New(
			uri.WithHTTPHandler(),
			uri.WithUserAkSk(en.cfg.Fetcher.AK, en.cfg.Fetcher.SK, en.cfg.Fetcher.RsHost),
		)
	} else {
		c = uri.New(uri.WithHTTPHandler())
	}
	resp, err := c.Get(ctx, uri.Request{URI: url})
	if err != nil {
		return
	}
	defer resp.Body.Close()
	stat, err := os.Stat(local)
	xl.Debugf("stat", stat, err, resp.Size)
	if err == nil && stat.Size() == resp.Size {
		return
	}
	file, err := os.Create(local)
	if err != nil {
		return
	}
	defer file.Close()
	n, err := io.Copy(file, resp.Body)
	xl.Debugf("copy", n)
	return
}

func (en TensorrtEvalNet) initNet(ctx context.Context) (err error) {
	xl := xlog.FromContextSafe(ctx)
	if en.cfg.TarFile != "" {
		tarFiles := make(map[string]string)
		tarFile := path.Join(en.cfg.Workspace, "model.tar")
		xl.Info("fetchFile start", en.cfg.TarFile, "to", tarFile)
		if err = en.fetchFile(ctx, en.cfg.TarFile, tarFile); err != nil {
			return errors.Wrap(err, "fetch tar file")
		}
		xl.Info("fetchFile end", en.cfg.TarFile)
		tarFiles, err = util.ExtractTar(en.cfg.Workspace, tarFile, false)
		if err != nil {
			return errors.Wrap(err, "extractTar")
		}
		xl.Info("extractTar success", tarFile, tarFiles)
		en.cfg.TarFiles = tarFiles
	}
	{
		for name, value := range en.cfg.CustomFiles {
			local := path.Join(en.cfg.Workspace, name)
			if err = en.fetchFile(ctx, value, local); err != nil {
				xl.Errorf("fetch custom file failed. %s %v", name, err)
				return
			}
			en.cfg.CustomFiles[name] = local
		}
	}
	xl.Info("initNet success")
	return
}

// NewTensorrtEvalNet ...
func NewTensorrtEvalNet(ctx context.Context, cfg *EvalConfig, tensorType string) (en *TensorrtEvalNet, err error) {
	en = &TensorrtEvalNet{
		cfg:        cfg,
		tensorType: tensorType,
	}
	if err = en.initNet(ctx); err != nil {
		return
	}
	switch en.tensorType {
	case tensorClassify:
		if err = tensor.ClassifierInit(ctx, cfg.TarFiles, cfg.BatchSize); err != nil {
			return
		}
	case tensorDetect:
		if err = tensor.DetecterInit(ctx, cfg.TarFiles, cfg.BatchSize); err != nil {
			return
		}
	default:
		err = ErrInvalidTensorType
		return
	}
	return
}

// NewTensorrtEvalNetLocal ...
func NewTensorrtEvalNetLocal(ctx context.Context, cfg *EvalConfig, tensorType string) (en *TensorrtEvalNet, err error) {
	en = &TensorrtEvalNet{
		cfg:        cfg,
		tensorType: tensorType,
	}
	switch en.tensorType {
	case tensorClassify:
		if err = tensor.ClassifierInit(ctx, cfg.TarFiles, cfg.BatchSize); err != nil {
			return
		}
	case tensorDetect:
		if err = tensor.DetecterInit(ctx, cfg.TarFiles, cfg.BatchSize); err != nil {
			return
		}
	default:
		err = ErrInvalidTensorType
		return
	}
	return
}

func (en TensorrtEvalNet) eval(ctx context.Context, reqs interface{},
) (resps []EvalResponseInner, err error) {

	reqsWithReqid := struct {
		Reqid string      `json:"reqid,omitempty"`
		Reqs  interface{} `json:"reqs"`
	}{
		Reqid: xlog.FromContextSafe(ctx).ReqId(),
		Reqs:  reqs,
	}
	reqsStr, _ := json.Marshal(reqsWithReqid)
	var ret interface{}
	switch en.tensorType {
	case tensorClassify:
		ret, err = tensor.Classify(ctx, string(reqsStr))
	case tensorDetect:
		ret, err = tensor.Detect(ctx, string(reqsStr))
	}
	if err != nil {
		return
	}

	var nr netResult
	if err = json.Unmarshal([]byte(ret.(string)), &nr); err != nil {
		err = errors.New("umarshal results err:" + err.Error())
		return
	}
	xl := xlog.FromContextSafe(ctx)
	if len(nr.Headers.Xlog) > 0 {
		xl.Xput(nr.Headers.Xlog)
	}
	for _, r := range nr.Results {
		if r.Code != 0 {
			xl.Errorf("net eval failed, code: (%d), error: (%s)", r.Code, r.Message)
			// if err = en.detectError(r.Code, r.Message); err != nil {
			// 	return
			// }
		}
		result := EvalResponseInner{Code: r.Code, Message: r.Message, Header: r.Header}
		if r.Result != "" {
			if err = json.Unmarshal([]byte(r.Result), &result.Result); err != nil {
				err = errors.New("umarshel result err:" + err.Error())
				return
			}
		}
		if r.ResultFile != "" {
			result.Stream = NewFile(_SchemeFilePrefix + r.ResultFile)
		}
		resps = append(resps, result)
	}
	return
}

// Eval ...
func (en TensorrtEvalNet) Eval(ctx context.Context, reqs []model.EvalRequestInner,
) (resps []EvalResponseInner, err error) {
	return en.eval(ctx, reqs)
}

// GroupEval ...
func (en TensorrtEvalNet) GroupEval(ctx context.Context, reqs []model.GroupEvalRequestInner,
) (resps []EvalResponseInner, err error) {
	return en.eval(ctx, reqs)
}

func (en *TensorrtEvalNet) preEval(ctx context.Context, req, resp interface{}) (err error) {
	resp = req
	return
}

func (en *TensorrtEvalNet) PreEval(ctx context.Context, req model.EvalRequestInner) (
	model.EvalRequestInner, error) {

	if !en.cfg.PreProcess.On {
		return req, nil
	}
	var resp model.EvalRequestInner
	err := en.preEval(ctx, req, &resp)
	return resp, err
}

func (en *TensorrtEvalNet) PreGroupEval(ctx context.Context, req model.GroupEvalRequestInner) (
	model.GroupEvalRequestInner, error) {
	if !en.cfg.PreProcess.On {
		return req, nil
	}
	var resp model.GroupEvalRequestInner
	err := en.preEval(ctx, req, &resp)
	return resp, err
}

func (en TensorrtEvalNet) Metrics() (string, error) {
	return "", nil
}
