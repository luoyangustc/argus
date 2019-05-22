package eval

import (
	"context"
	"encoding/json"
	"io"
	"io/ioutil"
	"net/http"
	"os"
	"path"
	"strings"

	"github.com/pkg/errors"
	xlog "github.com/qiniu/xlog.v1"

	"qiniu.com/argus/atserving/model"
	"qiniu.com/argus/com/cert"
	"qiniu.com/argus/com/py"
	"qiniu.com/argus/com/uri"
	"qiniu.com/argus/com/util"
	STS "qiniu.com/argus/sts/client"
)

const (
	evalModule            = "evals"
	netInitFunction       = "net_init"       // net_init(args...)
	netPreprocessFunction = "net_preprocess" // net_preprocess(model, args...)
	netInferenceFunction  = "net_inference"  // net_inference(model, args...)
)

var (
	ErrNoTarFile        = errors.New("eval net error: no tar_file was set")
	ErrNoImageWidth     = errors.New("eval net error: no image_width was set")
	ErrInvalidUseDevice = errors.New("eval net error: invalid use device")
)

// EvalConfig ...
type EvalConfig struct {
	App         string      `json:"app"`
	TarFile     string      `json:"tar_file"`
	ModelParams interface{} `json:"model_params"`

	ImageWidth    int    `json:"image_width"`
	MaxConcurrent int    `json:"max_concurrent,omitempty"`
	BatchSize     int    `json:"batch_size,omitempty"`
	UseDevice     string `json:"use_device"`

	CustomFiles  map[string]string      `json:"custom_files,omitempty"`
	CustomValues map[string]interface{} `json:"custom_values,omitempty"`

	TarFiles  map[string]string `json:"tar_files"`
	Workspace string            `json:"workspace"`

	Fetcher struct {
		UID    uint32
		AK     string
		SK     string
		RsHost string
	} `json:"-"`
	PreProcess struct {
		On      bool `json:"on"`
		Threads int  `json:"threads"`
	} `json:"process_pre"`
}

var _ Core = &EvalNet{}

// EvalNet ...
type EvalNet struct {
	cfg *EvalConfig
	sts STS.Client
	md  *py.PyModule
	net interface{}

	preProcess struct {
		threads py.PyThreads
	}
}

func (en EvalNet) fetchFile(ctx context.Context, url, local string) (err error) {
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
	// 先这么着
	if strings.HasPrefix(url, "proxy_http://") ||
		strings.HasPrefix(url, "proxy_https://") ||
		strings.HasPrefix(url, "proxy_qiniu://") {
		url, _ = improveURI(strings.TrimPrefix(url, "proxy_"), en.cfg.Fetcher.UID)
		var reader io.ReadCloser
		reader, _, _, err = en.sts.Get(ctx, url, nil, STS.WithOnlyProxy())
		if err != nil {
			xl.Warnf("open file failed. %s %v", strings.TrimPrefix(url, "proxy_"), err)
			return
		}
		defer reader.Close()
		var dest *os.File
		dest, err = os.Create(local)
		if err != nil {
			xl.Warnf("create file failed. %s %v", local, err)
			return
		}
		defer dest.Close()
		_, err = io.Copy(dest, reader)
		return
	}

	var c uri.Handler
	if strings.HasPrefix(url, "https://") {
		c = uri.New(
			uri.WithCertHTTPSHandler(cert.CACert, cert.ClientCert, cert.ClientKey),
		)
	} else if len(en.cfg.Fetcher.AK) > 0 {
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

func (en EvalNet) initNet(ctx context.Context) (err error) {
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

// NewEvalNet ...
func NewEvalNet(ctx context.Context, cfg *EvalConfig, sts STS.Client) (en *EvalNet, err error) {
	en = &EvalNet{
		cfg: cfg,
		sts: sts,
	}
	// pwd, _ := os.Getwd() // TODO 统一配置
	if err = en.initNet(ctx); err != nil {
		return
	}
	if en.md, err = py.NewPyModule(evalModule); err != nil {
		return
	}
	cfgStr, _ := json.Marshal(*en.cfg)
	if err = en.md.NetInit(netInitFunction, string(cfgStr)); err != nil {
		return
	}
	if cfg.PreProcess.On {
		threads := cfg.PreProcess.Threads
		if threads == 0 {
			threads = 30
		}
		en.preProcess = struct {
			threads py.PyThreads
		}{
			threads: py.NewPyThreads(threads),
		}
	}
	return
}

// NewEvalNetLocal ...
func NewEvalNetLocal(ctx context.Context, cfg *EvalConfig) (en *EvalNet, err error) {
	en = &EvalNet{
		cfg: cfg,
	}
	if en.md, err = py.NewPyModule(evalModule); err != nil {
		return
	}
	cfgStr, _ := json.Marshal(*en.cfg)
	if err = en.md.NetInit(netInitFunction, string(cfgStr)); err != nil {
		return
	}
	return
}

type netSingleResult struct {
	Result struct {
		Code    int    `json:"code"`
		Message string `json:"message"`
		Result  string `json:"result,omitempty"`
	} `json:"result"`
	Error string `json:"error"`
}

type netResult struct {
	Headers struct {
		Xlog     []string               `json:"xlog"`
		Xstatus  string                 `json:"xstatus"`
		Xmetrics map[string]interface{} `json:"xmetrics"`
	} `json:"headers"`
	Results []struct {
		Code       int         `json:"code"`
		Message    string      `json:"message"`
		Header     http.Header `json:"header,omitempty"`
		Result     string      `json:"result,omitempty"`
		ResultFile string      `json:"result_file,omitempty"`
	} `json:"results"`
	Error string `json:"error"`
}

func (en EvalNet) eval(ctx context.Context, reqs interface{},
) (resps []EvalResponseInner, err error) {

	reqsWithReqid := struct {
		Reqid string      `json:"reqid,omitempty"`
		Reqs  interface{} `json:"reqs"`
	}{
		Reqid: xlog.FromContextSafe(ctx).ReqId(),
		Reqs:  reqs,
	}
	reqsStr, _ := json.Marshal(reqsWithReqid)
	ret, err := en.md.MdRun(netInferenceFunction, string(reqsStr))
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
func (en EvalNet) Eval(ctx context.Context, reqs []model.EvalRequestInner,
) (resps []EvalResponseInner, err error) {
	return en.eval(ctx, reqs)
}

// GroupEval ...
func (en EvalNet) GroupEval(ctx context.Context, reqs []model.GroupEvalRequestInner,
) (resps []EvalResponseInner, err error) {
	return en.eval(ctx, reqs)
}

func (en *EvalNet) preEval(ctx context.Context, req, resp interface{}) (err error) {

	reqWithReqid := struct {
		Reqid string      `json:"reqid,omitempty"`
		Req   interface{} `json:"req"`
	}{
		Reqid: xlog.FromContextSafe(ctx).ReqId(),
		Req:   req,
	}
	reqStr, _ := json.Marshal(reqWithReqid)
	ret, err := en.md.MdRunWithThreads(netPreprocessFunction, string(reqStr), en.preProcess.threads)
	if err != nil {
		return
	}

	var nr netSingleResult
	if err = json.Unmarshal([]byte(ret.(string)), &nr); err != nil {
		err = errors.New("umarshal results err:" + err.Error())
		return
	}
	xl := xlog.FromContextSafe(ctx)
	if nr.Result.Code != 0 {
		xl.Errorf("net preprocess failed, code: (%d), error: (%s)", nr.Result.Code, nr.Result.Message)
		if err = detectEvalError(nr.Result.Code, nr.Result.Message); err != nil {
			return
		}
	}
	err = json.Unmarshal([]byte(nr.Result.Result), resp)
	return
}

func (en *EvalNet) PreEval(ctx context.Context, req model.EvalRequestInner) (
	model.EvalRequestInner, error) {
	if !en.cfg.PreProcess.On {
		return req, nil
	}
	var resp model.EvalRequestInner
	switch v := req.Data.URI.(type) {
	case model.BYTES:
		file := path.Join(en.cfg.Workspace, xlog.GenReqId())
		_ = ioutil.WriteFile(file, v.Bytes(), 0755)
		req.Data.URI = file
	}
	err := en.preEval(ctx, req, &resp)
	return resp, err
}

func (en *EvalNet) PreGroupEval(ctx context.Context, req model.GroupEvalRequestInner) (
	model.GroupEvalRequestInner, error) {

	if !en.cfg.PreProcess.On {
		return req, nil
	}
	var resp model.GroupEvalRequestInner
	for i, data := range req.Data {
		switch v := data.URI.(type) {
		case model.BYTES:
			file := path.Join(en.cfg.Workspace, xlog.GenReqId())
			_ = ioutil.WriteFile(file, v.Bytes(), 0755)
			req.Data[i].URI = file
		}
	}
	err := en.preEval(ctx, req, &resp)
	return resp, err
}

func (en EvalNet) Metrics() (string, error) {
	return en.md.Metrics()
}
