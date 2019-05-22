package inference

/*
#cgo LDFLAGS: -ldl
#include <dlfcn.h>
#include <limits.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>

typedef void (*INIT_FUNC) ( \
	void*, const int, \
	int*, char**);
typedef void* (*CREATE_FUNC)( \
	void*, const int, \
	int*, char**);
typedef void (*PREPROCESS_FUNC)( \
	const void*, void*, const int, \
	int*, char**, void*, int*);
typedef void (*INFERENCE_FUNC)(\
	const void*, void*, const int, \
	int*, char**, void*, int*);

static uintptr_t pluginOpen(const char* path, char** err) {
	void* h = dlopen(path, RTLD_NOW|RTLD_GLOBAL);
	if (h == NULL) {
		*err = (char*)dlerror();
	}
	return (uintptr_t)h;
}

static INIT_FUNC lookupInitFunc(uintptr_t h, char** err) {
	void* r = dlsym((void*)h, "initEnv");
	if (r == NULL) {
		*err = (char*)dlerror();
	}
	return (INIT_FUNC)r;
}
void callInitFunc(INIT_FUNC f,
	void* args, const int args_size,
	int* code, char** err) {
	f(args, args_size, code, err);
	return;
}

static CREATE_FUNC lookupCreateFunc(uintptr_t h, char** err) {
	void* r = dlsym((void*)h, "createNet");
	if (r == NULL) {
		*err = (char*)dlerror();
	}
	return (CREATE_FUNC)r;
}
void* callCreateFunc(CREATE_FUNC f,
	void* args, const int args_size,
	int* code, char** err) {
	return f(args, args_size, code, err);
}

static PREPROCESS_FUNC lookupPreprocessFunc(uintptr_t h, char** err) {
	void* r = dlsym((void*)h, "netPreprocess");
	if (r == NULL) {
		*err = (char*)dlerror();
	}
	return (PREPROCESS_FUNC)r;
}
void callPreprocessFunc(PREPROCESS_FUNC f,
	const void* net, void* args, const int args_size,
	int* code, char** err, void* ret, int* ret_size) {
	return f(net, args, args_size, code, err, ret, ret_size);
}

static INFERENCE_FUNC lookupInferenceFunc(uintptr_t h, char** err) {
	void* r = dlsym((void*)h, "netInference");
	if (r == NULL) {
		*err = (char*)dlerror();
	}
	return (INFERENCE_FUNC)r;
}
void callInferenceFunc(INFERENCE_FUNC f,
	const void* net, void* args, int args_size,
	int* code, char** err, void* ret, int* ret_size) {
	return f(net, args, args_size, code, err, ret, ret_size);
}

*/
import "C"
import (
	"context"
	"crypto/sha1"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io/ioutil"
	"path/filepath"
	"runtime"
	"strings"
	"sync"
	"unsafe"

	"github.com/gogo/protobuf/proto"

	"github.com/qiniu/http/httputil.v1"
	"github.com/qiniu/xlog.v1"

	"qiniu.com/argus/atserving/model"
	lib "qiniu.com/argus/serving_eval/inference/lib"
)

var _ Creator = &Lib{}

type Lib struct {
	Workspace string

	Path   string
	plugin C.uintptr_t

	initFUNC       C.INIT_FUNC
	createFUNC     C.CREATE_FUNC
	preprocessFUNC C.PREPROCESS_FUNC
	inferenceFUNC  C.INFERENCE_FUNC

	initialized bool
	*sync.Mutex
}

func NewLib(ctx context.Context, workspace, path string) (*Lib, error) {
	var (
		xl = xlog.FromContextSafe(ctx)

		cPath    = make([]byte, C.PATH_MAX+1)
		cRelName = make([]byte, len(path)+1)
	)

	copy(cRelName, path)
	if C.realpath(
		(*C.char)(unsafe.Pointer(&cRelName[0])),
		(*C.char)(unsafe.Pointer(&cPath[0])),
	) == nil {
		err := errors.New(`plugin.Open("` + path + `"): realpath failed`)
		xl.Errorf("%v", err)
		return nil, err
	}

	var (
		cErr *C.char
	)
	h := C.pluginOpen((*C.char)(unsafe.Pointer(&cPath[0])), &cErr)
	if h == 0 {
		err := errors.New(`plugin.Open("` + path + `"): ` + C.GoString(cErr))
		xl.Errorf("%v", err)
		return nil, err
	}

	lib := &Lib{Workspace: workspace, Path: path, plugin: h, Mutex: new(sync.Mutex)}
	lib.initFUNC = C.lookupInitFunc(lib.plugin, &cErr)
	if lib.initFUNC == nil {
		err := fmt.Errorf("function init: %s", C.GoString(cErr))
		xl.Errorf("%v", err)
		return lib, err
	}
	lib.createFUNC = C.lookupCreateFunc(lib.plugin, &cErr)
	if lib.createFUNC == nil {
		err := fmt.Errorf("function create: %s", C.GoString(cErr))
		xl.Errorf("%v", err)
		return lib, err
	}
	// 这个接口设计不完整，先全部禁止
	// lib.preprocessFUNC = C.lookupPreprocessFunc(lib.plugin, &cErr)
	// if lib.preprocessFUNC == nil {
	// 	err := fmt.Errorf("function preprocess: %s", C.GoString(cErr))
	// 	xl.Errorf("%v", err)
	// 	// return lib, err
	// }
	lib.inferenceFUNC = C.lookupInferenceFunc(lib.plugin, &cErr)
	if lib.inferenceFUNC == nil {
		err := fmt.Errorf("function inference: %s", C.GoString(cErr))
		xl.Errorf("%v", err)
		return lib, err
	}

	return lib, nil
}

func (_lib *Lib) Create(ctx context.Context, params *CreateParams) (Instance, error) {

	var (
		// xl = xlog.FromContextSafe(ctx)

		_params = &lib.CreateParams{
			UseDevice: proto.String(params.UseDevice),
			BatchSize: proto.Int32(int32(params.BatchSize)),
			Env: &lib.CreateParams_Env{
				App:       proto.String(params.App),
				Workspace: proto.String(params.Workspace),
			},
		}
	)
	{
		for name, file := range params.ModelFiles {
			bs, _ := ioutil.ReadFile(file)
			_params.ModelFiles = append(
				_params.ModelFiles,
				&lib.CreateParams_File{
					Name: proto.String(name),
					Body: bs,
				})
		}
	}
	if params.ModelParams != nil {
		bs, _ := json.Marshal(params.ModelParams)
		_params.ModelParams = proto.String(string(bs))
	}
	{
		for name, file := range params.CustomFiles {
			bs, _ := ioutil.ReadFile(file)
			_params.CustomFiles = append(
				_params.CustomFiles,
				&lib.CreateParams_File{
					Name: proto.String(name),
					Body: bs,
				})
		}
	}
	if params.CustomParams != nil {
		bs, _ := json.Marshal(params.CustomParams)
		_params.CustomParams = proto.String(string(bs))
	}

	return newLibInstance(_lib, _params)
}

var _ Instance = &LibInstance{}

type LibInstance struct {
	*Lib
	ctx unsafe.Pointer

	ch chan struct {
		done chan bool
		f    func()
	}
}

func newLibInstance(_lib *Lib, params *lib.CreateParams) (*LibInstance, error) {

	i := &LibInstance{
		Lib: _lib,
		ch: make(chan struct {
			done chan bool
			f    func()
		}),
	}

	var (
		xl   = xlog.NewDummy()
		wait sync.WaitGroup

		createParams, _ = proto.Marshal(params)
		cCode           C.int
		cErr            *C.char
	)

	wait.Add(1)
	go func(ctx context.Context) {
		runtime.LockOSThread()

		func() {
			i.Lock()
			defer i.Unlock()

			if i.initialized {
				return
			}
			_params := &lib.InitParams{App: params.Env.App}
			bs, _ := proto.Marshal(_params)
			_CBs := C.CBytes(bs)
			defer C.free(_CBs)
			C.callInitFunc(i.initFUNC, _CBs, C.int(len(bs)), &cCode, &cErr)
			C.fflush(C.stdout)
			C.fflush(C.stderr)
		}()

		_CCreateParams := C.CBytes(createParams)
		defer C.free(_CCreateParams)
		i.ctx = C.callCreateFunc(_lib.createFUNC,
			_CCreateParams, C.int(len(createParams)),
			&cCode, &cErr)
		C.fflush(C.stdout)
		C.fflush(C.stderr)

		wait.Done()

		for r := range i.ch {
			func() {
				defer func() {
					if err := recover(); err != nil {
						// TODO
					}
					r.done <- true
				}()
				r.f()
			}()
		}
	}(xlog.NewContext(
		context.Background(),
		xlog.NewWith("instance."+xlog.GenReqId()),
	))

	wait.Wait()
	if int(cCode) != 0 && int(cCode) != 200 {
		xl.Errorf("%d %s", int(cCode), C.GoString(cErr))
		return nil, httputil.NewError(int(cCode), C.GoString(cErr))
	}

	return i, nil
}

func (i *LibInstance) preprocess(
	ctx context.Context, request *lib.InferenceRequest,
) (*lib.InferenceRequest, error) {

	var (
		xl          = xlog.FromContextSafe(ctx)
		_request, _ = proto.Marshal(request)

		retSize C.int
		ret     = C.malloc(1024 * 1024 * 4)
		cCode   C.int
		cErr    *C.char
	)
	defer C.free(ret)
	i.invoke(func() {
		_CRequest := C.CBytes(_request)
		defer C.free(_CRequest)
		C.callPreprocessFunc(
			i.preprocessFUNC, i.ctx,
			_CRequest, C.int(len(_request)),
			&cCode, &cErr,
			ret, &retSize,
		)
		C.fflush(C.stdout)
		C.fflush(C.stderr)
	})
	if int(cCode) != 0 && int(cCode) != 200 {
		xl.Errorf("%d %s", int(cCode), C.GoString(cErr))
		return nil, httputil.NewError(int(cCode), C.GoString(cErr))
	}

	var (
		bs       = C.GoBytes(ret, retSize)
		response = &lib.InferenceRequest{}
	)
	if err := proto.Unmarshal(bs, response); err != nil {
		xl.Errorf("parse preprocess response failed. %v", err)
		return nil, err
	}
	return response, nil
}

func (i *LibInstance) Preprocess(
	ctx context.Context, req model.EvalRequestInner,
) (model.EvalRequestInner, error) {

	if i.preprocessFUNC == nil {
		return req, nil
	}

	var (
		request = &lib.InferenceRequest{
			Data: &lib.InferenceRequest_RequestData{},
		}
	)
	{
		switch v := req.Data.URI.(type) {
		case model.BYTES:
			request.Data.Uri = proto.String("")
			request.Data.Body = v
		case model.STRING:
			bs, _ := ioutil.ReadFile(v.String())
			request.Data.Uri = proto.String(v.String())
			request.Data.Body = bs
		}
		if req.Data.Attribute != nil {
			bs, _ := json.Marshal(req.Data.Attribute)
			request.Data.Attribute = proto.String(string(bs))
		}
	}
	if req.Params != nil {
		bs, _ := json.Marshal(req.Params)
		request.Params = proto.String(string(bs))
	}

	response, err := i.preprocess(ctx, request)
	if err != nil {
		return model.EvalRequestInner{}, err
	}

	resp := model.EvalRequestInner{}
	resp.Data.URI = response.Data.GetBody()
	if attribute := response.Data.GetAttribute(); attribute != "" {
		_ = json.Unmarshal([]byte(attribute), &resp.Data.Attribute)
	}
	if params := response.GetParams(); params != "" {
		_ = json.Unmarshal([]byte(params), &resp.Params)
	}

	return resp, nil
}

func (i *LibInstance) PreprocessGroup(
	ctx context.Context, req model.GroupEvalRequestInner,
) (model.GroupEvalRequestInner, error) {

	if i.preprocessFUNC == nil {
		return req, nil
	}

	var (
		request = &lib.InferenceRequest{
			Datas: make([]*lib.InferenceRequest_RequestData, 0, len(req.Data)),
		}
	)
	{
		for _, data := range req.Data {
			var _data *lib.InferenceRequest_RequestData
			switch v := data.URI.(type) {
			case model.BYTES:
				_data = &lib.InferenceRequest_RequestData{Body: v}
			case model.STRING:
				bs1, _ := ioutil.ReadFile(v.String())
				_data = &lib.InferenceRequest_RequestData{
					Uri:  proto.String(v.String()),
					Body: bs1,
				}
			}
			if data.Attribute != nil {
				bs2, _ := json.Marshal(data.Attribute)
				_data.Attribute = proto.String(string(bs2))
			}
			request.Datas = append(request.Datas, _data)
		}
	}
	if req.Params != nil {
		bs, _ := json.Marshal(req.Params)
		request.Params = proto.String(string(bs))
	}

	response, err := i.preprocess(ctx, request)
	if err != nil {
		return model.GroupEvalRequestInner{}, err
	}

	resp := model.GroupEvalRequestInner{
		Data: make([]model.ResourceInner, 0, len(response.GetDatas())),
	}
	{
		for _, data := range response.GetDatas() {
			_data := model.ResourceInner{}
			_data.URI = data.GetBody()
			if attribute := data.GetAttribute(); attribute != "" {
				_ = json.Unmarshal([]byte(attribute), &_data.Attribute)
			}
			resp.Data = append(resp.Data, _data)
		}
	}
	if params := response.GetParams(); params != "" {
		_ = json.Unmarshal([]byte(params), &resp.Params)
	}

	return resp, nil
}

func (i *LibInstance) invoke(f func()) {
	done := make(chan bool)
	i.ch <- struct {
		done chan bool
		f    func()
	}{
		done: done,
		f:    f,
	}
	<-done
}

func (i *LibInstance) inference(
	ctx context.Context, requests *lib.InferenceRequests,
) ([]Response, error) {

	var (
		xl           = xlog.FromContextSafe(ctx)
		_requests, _ = proto.Marshal(requests)

		retSize C.int
		ret     = C.malloc(1024 * 1024 * 4)
		cCode   C.int
		cErr    *C.char
	)
	defer C.free(ret)

	i.invoke(func() {
		_CRequests := C.CBytes(_requests)
		defer C.free(_CRequests)
		C.callInferenceFunc(
			i.inferenceFUNC, i.ctx,
			_CRequests, C.int(len(_requests)),
			&cCode, &cErr,
			ret, &retSize,
		)
		C.fflush(C.stdout)
		C.fflush(C.stderr)
	})

	if int(cCode) != 0 && int(cCode) != 200 {
		xl.Errorf("%d %s", int(cCode), C.GoString(cErr))
		code, message := foramtCodeMessage(int(cCode), C.GoString(cErr))
		return nil, httputil.NewError(code, message)
	}

	var (
		bs       = C.GoBytes(ret, retSize)
		response = &lib.InferenceResponses{}
	)
	if err := proto.Unmarshal(bs, response); err != nil {
		xl.Errorf("parse inference response failed. %v", err)
		return nil, err
	}

	var (
		responses = make([]Response, 0, len(response.GetResponses()))
	)
	for _, resp := range response.GetResponses() {
		code, message := foramtCodeMessage(int(resp.GetCode()), resp.GetMessage())
		_resp := Response{
			Code:    code,
			Message: message,
		}

		if body := resp.GetBody(); body != nil && len(body) > 0 {
			_resp.Body = body
		} else {
			err := json.Unmarshal([]byte(resp.GetResult()), &_resp.Result)
			if err != nil {
				xl.Errorf("json.Unmarshal error:%v, result:%v", err, resp.GetResult())
				return nil, err
			}
		}
		responses = append(responses, _resp)
	}
	return responses, nil
}

func (i *LibInstance) Inference(
	ctx context.Context, reqs []model.EvalRequestInner,
) ([]Response, error) {

	var (
		requests = &lib.InferenceRequests{
			Requests: make([]*lib.InferenceRequest, 0, len(reqs)),
		}
	)
	for _, req := range reqs {
		request := &lib.InferenceRequest{
			Data: &lib.InferenceRequest_RequestData{},
		}
		{
			switch v := req.Data.URI.(type) {
			case model.BYTES:
				request.Data.Body = v
			case model.STRING:
				bs, _ := ioutil.ReadFile(v.String())
				request.Data.Body = bs
			}
			if req.Data.Attribute != nil {
				bs, _ := json.Marshal(req.Data.Attribute)
				request.Data.Attribute = proto.String(string(bs))
			}
		}
		if req.Params != nil {
			bs, _ := json.Marshal(req.Params)
			request.Params = proto.String(string(bs))
		}
		requests.Requests = append(requests.Requests, request)
	}

	return i.inference(ctx, requests)
}

func (i *LibInstance) InferenceGroup(
	ctx context.Context, reqs []model.GroupEvalRequestInner,
) ([]Response, error) {

	var (
		requests = &lib.InferenceRequests{
			Requests: make([]*lib.InferenceRequest, 0, len(reqs)),
		}
	)
	for _, req := range reqs {
		request := &lib.InferenceRequest{}
		{
			for _, data := range req.Data {
				data2 := &lib.InferenceRequest_RequestData{}
				switch v := data.URI.(type) {
				case model.BYTES:
					data2.Body = v
				case model.STRING:
					bs1, _ := ioutil.ReadFile(v.String())
					data2.Body = bs1
				}
				if data.Attribute != nil {
					bs2, _ := json.Marshal(data.Attribute)
					data2.Attribute = proto.String(string(bs2))
				}
				request.Datas = append(request.Datas, data2)
			}
		}
		if req.Params != nil {
			bs, _ := json.Marshal(req.Params)
			request.Params = proto.String(string(bs))
		}
		requests.Requests = append(requests.Requests, request)
	}

	return i.inference(ctx, requests)
}

func (i *LibInstance) newFilename(uri string) string {
	sum := sha1.Sum([]byte(strings.Join([]string{xlog.GenReqId(), uri}, "_")))
	return filepath.Join(i.Workspace, hex.EncodeToString(sum[:]))
}
