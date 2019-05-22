package eval

import (
	"context"
	"io"
	"io/ioutil"
	"net/http"
	"os"
	"path"
	"strings"

	"github.com/pkg/errors"
	xlog "github.com/qiniu/xlog.v1"

	"qiniu.com/argus/atserving/model"
	"qiniu.com/argus/com/uri"
	"qiniu.com/argus/com/util"
	"qiniu.com/argus/serving_eval/inference"
	STS "qiniu.com/argus/sts/client"
)

// EvalResponseInner 内部使用的返回结果
type EvalResponseInner struct {
	Code    int         `json:"code"`
	Message string      `json:"message"`
	Header  http.Header `json:"header,omitempty"`
	Result  interface{} `json:"result,omitempty"`
	// URI     *string     `json:"uri,omitempty"`
	Stream Stream `json:"-"`
	// Data    io.ReadCloser `json:"-"`
}

// Core ...
// 如果返回为二进制内容，则返回[]string（表示本地文件名），否则返回空数组
// 可能出现某些返回name，某些返回Response的情况，比如个别处理错误的情况，所以需要逐个判断
type Core interface {
	Eval(context.Context, []model.EvalRequestInner) ([]EvalResponseInner, error)
	GroupEval(context.Context, []model.GroupEvalRequestInner) ([]EvalResponseInner, error)

	PreEval(context.Context, model.EvalRequestInner) (model.EvalRequestInner, error)
	PreGroupEval(context.Context, model.GroupEvalRequestInner) (model.GroupEvalRequestInner, error)
}

//----------------------------------------------------------------------------//

var _ Core = MockCore{}

// MockCore JUST FOR TEST
type MockCore struct{}

// NewMockCore ...
func NewMockCore() MockCore { return MockCore{} }

func (mock MockCore) PreEval(ctx context.Context, req model.EvalRequestInner) (
	model.EvalRequestInner, error) {
	return req, nil
}

func (mock MockCore) PreGroupEval(ctx context.Context, req model.GroupEvalRequestInner) (
	model.GroupEvalRequestInner, error) {
	return req, nil
}

// Eval 返回原内容
func (mock MockCore) Eval(ctx context.Context, reqs []model.EvalRequestInner,
) ([]EvalResponseInner, error) {

	var resps = make([]EvalResponseInner, 0, len(reqs))
	for i, n := 0, len(reqs); i < n; i++ {
		stream, err := func() (stream Stream, err error) {
			switch v := reqs[i].Data.URI.(type) {
			case model.BYTES:
				return NewMem(v), nil
			case model.STRING:
				bs, err := ioutil.ReadFile(v.String())
				if err != nil {
					return nil, errors.Wrap(err, "open src file")
				}
				return NewMem(bs), nil
			default:
				return nil, nil
			}
		}()
		if err != nil {
			return nil, errors.Wrap(err, "hello eval ")
		}
		resps = append(resps, EvalResponseInner{
			Header: http.Header{
				model.XOriginA: []string{"MOCK:1"},
			},
			Stream: stream,
		})
	}
	return resps, nil
}

// GroupEval 返回各资源大小
func (mock MockCore) GroupEval(ctx context.Context, reqs []model.GroupEvalRequestInner) (
	[]EvalResponseInner, error) {
	var (
		resps = make([]EvalResponseInner, 0, len(reqs))
	)
	for i, n := 0, len(reqs); i < n; i++ {
		var result = make([]int64, 0, len(reqs[i].Data))
		for _, data := range reqs[i].Data {
			switch v := data.URI.(type) {
			case model.BYTES:
				result = append(result, int64(len(v.Bytes())))
			case model.STRING:
				var (
					info os.FileInfo
					err  error
				)
				if info, err = os.Lstat(v.String()); err != nil {
					return nil, err
				}
				result = append(result, info.Size())
			}
		}
		resps = append(resps, EvalResponseInner{
			Header: http.Header{
				model.XOriginA: []string{"MOCK:1"},
			},
			Result: result,
		})
	}
	return resps, nil
}

//----------------------------------------------------------------------------//

var _ Core = &Inference{}

type Inference struct {
	*EvalConfig
	sts STS.Client

	inference.Creator
	inference.Instance
}

func (in Inference) fetchFile(ctx context.Context, url, local string) (err error) {
	xl := xlog.FromContextSafe(ctx)
	_ = os.MkdirAll(path.Dir(local), 0755)
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
		url, _ = improveURI(strings.TrimPrefix(url, "proxy_"), in.Fetcher.UID)
		var reader io.ReadCloser
		reader, _, _, err = in.sts.Get(ctx, url, nil, STS.WithOnlyProxy())
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
	if len(in.EvalConfig.Fetcher.AK) > 0 {
		c = uri.New(
			uri.WithHTTPHandler(),
			uri.WithUserAkSk(
				in.EvalConfig.Fetcher.AK,
				in.EvalConfig.Fetcher.SK,
				in.EvalConfig.Fetcher.RsHost,
			),
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

func (in Inference) initNet(ctx context.Context) (err error) {
	xl := xlog.FromContextSafe(ctx)
	if in.EvalConfig.TarFile != "" {
		tarFiles := make(map[string]string)
		tarFile := path.Join(in.EvalConfig.Workspace, "model.tar")
		xl.Info("fetchFile start", in.EvalConfig.TarFile, "to", tarFile)
		if err = in.fetchFile(ctx, in.EvalConfig.TarFile, tarFile); err != nil {
			return errors.Wrap(err, "fetch tar file")
		}
		xl.Info("fetchFile end", in.EvalConfig.TarFile)
		tarFiles, err = util.ExtractTar(in.EvalConfig.Workspace, tarFile, false)
		if err != nil {
			return errors.Wrap(err, "extractTar")
		}
		xl.Info("extractTar success", tarFile, tarFiles)
		in.EvalConfig.TarFiles = tarFiles
	}
	{
		for name, value := range in.EvalConfig.CustomFiles {
			local := path.Join(in.EvalConfig.Workspace, name)
			if err = in.fetchFile(ctx, value, local); err != nil {
				xl.Errorf("fetch custom file failed. %s %v", name, err)
				return
			}
			in.EvalConfig.CustomFiles[name] = local
		}
	}
	xl.Info("pre init net success")
	return
}

func NewInference(ctx context.Context, cfg *EvalConfig, sts STS.Client) (in *Inference, err error) {
	xl := xlog.FromContextSafe(ctx)
	in = &Inference{
		EvalConfig: cfg,
		sts:        sts,
	}
	if in.Creator, err = inference.NewLib(ctx, cfg.Workspace, "inference.so"); err != nil {
		return
	}
	if err = in.initNet(ctx); err != nil {
		return
	}
	if in.Instance, err = in.Creator.Create(ctx,
		&inference.CreateParams{
			App:           in.EvalConfig.App,
			Workspace:     in.EvalConfig.Workspace,
			UseDevice:     in.EvalConfig.UseDevice,
			BatchSize:     in.EvalConfig.BatchSize,
			MaxConcurrent: in.EvalConfig.MaxConcurrent,
			ModelFiles:    in.EvalConfig.TarFiles,
			ModelParams:   in.EvalConfig.ModelParams,
			CustomFiles:   in.EvalConfig.CustomFiles,
			CustomParams:  in.EvalConfig.CustomValues,
		},
	); err != nil {
		xl.Errorf("new inference failed. %v", err)
		return
	}
	xl.Info("new inference success")
	// if cfg.PreProcess.On {
	// 	threads := cfg.PreProcess.Threads
	// 	if threads == 0 {
	// 		threads = 30
	// 	}
	// 	en.preProcess = struct {
	// 		threads _PyThreads
	// 	}{
	// 		threads: newPyThreads(threads),
	// 	}
	// }
	return
}

func NewInferenceLocal(ctx context.Context, cfg *EvalConfig) (in *Inference, err error) {
	var creator inference.Creator
	if creator, err = inference.NewLib(ctx, cfg.Workspace, "inference.so"); err != nil {
		return
	}
	return NewInferenceDirect(ctx, cfg, creator)
}

func NewInferenceDirect(
	ctx context.Context, cfg *EvalConfig,
	creator inference.Creator,
) (in *Inference, err error) {
	in = &Inference{
		EvalConfig: cfg,
		Creator:    creator,
	}
	if err = in.initNet(ctx); err != nil {
		return
	}
	if in.Instance, err = in.Creator.Create(ctx,
		&inference.CreateParams{
			App:           in.EvalConfig.App,
			Workspace:     in.EvalConfig.Workspace,
			UseDevice:     in.EvalConfig.UseDevice,
			BatchSize:     in.EvalConfig.BatchSize,
			MaxConcurrent: in.EvalConfig.MaxConcurrent,
			ModelFiles:    in.EvalConfig.TarFiles,
			ModelParams:   in.EvalConfig.ModelParams,
			CustomFiles:   in.EvalConfig.CustomFiles,
			CustomParams:  in.EvalConfig.CustomValues,
		}); err != nil {
		return
	}
	return
}

func (in Inference) Eval(ctx context.Context, reqs []model.EvalRequestInner,
) (resps []EvalResponseInner, err error) {

	responses, err := in.Inference(ctx, reqs)
	if err != nil {
		return
	}
	for _, response := range responses {
		resp := EvalResponseInner{
			Code:    response.Code,
			Message: response.Message,
			Result:  response.Result,
		}
		if len(response.Body) > 0 {
			resp.Stream = NewMem(response.Body)
		}
		resps = append(resps, resp)
	}

	return
}

func (in Inference) GroupEval(ctx context.Context, reqs []model.GroupEvalRequestInner,
) (resps []EvalResponseInner, err error) {

	responses, err := in.InferenceGroup(ctx, reqs)
	if err != nil {
		return
	}
	for _, response := range responses {
		resp := EvalResponseInner{
			Code:    response.Code,
			Message: response.Message,
			Result:  response.Result,
		}
		if len(response.Body) > 0 {
			resp.Stream = NewMem(response.Body)
		}
		resps = append(resps, resp)
	}

	return
}

func (in Inference) PreEval(ctx context.Context, req model.EvalRequestInner) (
	model.EvalRequestInner, error) {

	if !in.EvalConfig.PreProcess.On {
		return req, nil
	}
	return in.Instance.Preprocess(ctx, req)
}

func (in Inference) PreGroupEval(ctx context.Context, req model.GroupEvalRequestInner) (
	model.GroupEvalRequestInner, error) {

	if !in.EvalConfig.PreProcess.On {
		return req, nil
	}
	return in.Instance.PreprocessGroup(ctx, req)
}

func (in Inference) Metrics() (string, error) {
	return "", nil
}
