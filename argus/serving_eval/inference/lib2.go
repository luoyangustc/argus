package inference

/*
#include <dlfcn.h>
#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

typedef void* PredictorContext;
typedef void* PredictorHandle;

typedef const char* (*FuncGetLastError)();

typedef int (*FuncCreate)(const void*, const int, PredictorContext*);

typedef int (*FuncHandle)(PredictorContext,
                          const void*, const int,
                          PredictorHandle*);

typedef int (*FuncInference)(PredictorHandle,
                             const void*, const int,
                             void**, int*);

typedef int (*FuncFree)(PredictorContext);

static uintptr_t pluginOpen(const char* path, char** err) {
  void* h = dlopen(path, RTLD_NOW | RTLD_GLOBAL);
  if (h == NULL) *err = (char*)dlerror();
  return (uintptr_t)h;
}

static FuncGetLastError lookupFuncGetLastError(uintptr_t h, char** err) {
  void* r = dlsym((void*)h, "QTGetLastError");
  if (r == NULL) *err = (char*)dlerror();
  return (FuncGetLastError)r;
}
void callFuncGetLastError(FuncGetLastError f, const char** err) { *err = f(); }

static FuncCreate lookupFuncCreate(uintptr_t h, char** err) {
  void* r = dlsym((void*)h, "QTPredCreate");
  if (r == NULL) *err = (char*)dlerror();
  return (FuncCreate)r;
}
int callFuncCreate(FuncCreate f,
                   const void* params_data, const int params_size,
                   PredictorContext* ctx) {
  return f(params_data, params_size, ctx);
}

static FuncHandle lookupFuncHandle(uintptr_t h, char** err) {
  void* r = dlsym((void*)h, "QTPredHandle");
  if (r == NULL) *err = (char*)dlerror();
  return (FuncHandle)r;
}
int callFuncHandle(FuncHandle f,
                   PredictorContext ctx,
                   const void* params_data, const int params_size,
                   PredictorHandle* handle) {
  return f(ctx, params_data, params_size, handle);
}

static FuncInference lookupFuncInference(uintptr_t h, char** err) {
  void* r = dlsym((void*)h, "QTPredInference");
  if (r == NULL) *err = (char*)dlerror();
  return (FuncInference)r;
}
int callFuncInference(FuncInference f,
                      PredictorHandle handle,
                      const void* in_data, const int in_size,
                      void** out_data, int* out_size) {
  return f(handle, in_data, in_size, out_data, out_size);
}

static FuncFree lookupFuncFree(uintptr_t h, char** err) {
  void* r = dlsym((void*)h, "QTPredFree");
  if (r == NULL) *err = (char*)dlerror();
  return (FuncFree)r;
}
int callFuncFree(FuncFree f, PredictorContext ctx) {
  return f(ctx);
}

*/
import "C"
import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io/ioutil"
	"runtime"
	"sync"
	"time"
	"unsafe"

	"github.com/gogo/protobuf/proto"

	httputil "github.com/qiniu/http/httputil.v1"
	xlog "github.com/qiniu/xlog.v1"

	"qiniu.com/argus/atserving/model"
	lib "qiniu.com/argus/serving_eval/inference/lib"
)

var _ Creator = &Lib{}

type Lib2 struct {
	Workspace string

	Path   string
	plugin C.uintptr_t

	getLastErrorFunc C.FuncGetLastError
	createFunc       C.FuncCreate
	handleFunc       C.FuncHandle
	inferenceFunc    C.FuncInference
	freeFunc         C.FuncFree
}

func NewLib2(ctx context.Context, workspace, path string) Creator {
	lib, err := newLib2(ctx, workspace, path)
	if err != nil {
		panic(err)
	}
	return lib
}

func newLib2(ctx context.Context, workspace, path string) (*Lib2, error) {
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

	lib := &Lib2{Workspace: workspace, Path: path, plugin: h}
	lib.getLastErrorFunc = C.lookupFuncGetLastError(lib.plugin, &cErr)
	if lib.getLastErrorFunc == nil {
		err := fmt.Errorf("function getLastError: %s", C.GoString(cErr))
		xl.Errorf("%v", err)
		return lib, err
	}
	lib.createFunc = C.lookupFuncCreate(lib.plugin, &cErr)
	if lib.createFunc == nil {
		err := fmt.Errorf("function create: %s", C.GoString(cErr))
		xl.Errorf("%v", err)
		return lib, err
	}
	lib.handleFunc = C.lookupFuncHandle(lib.plugin, &cErr)
	if lib.handleFunc == nil {
		err := fmt.Errorf("function handle: %s", C.GoString(cErr))
		xl.Errorf("%v", err)
		return lib, err
	}
	lib.inferenceFunc = C.lookupFuncInference(lib.plugin, &cErr)
	if lib.inferenceFunc == nil {
		err := fmt.Errorf("function inference: %s", C.GoString(cErr))
		xl.Errorf("%v", err)
		return lib, err
	}
	lib.freeFunc = C.lookupFuncFree(lib.plugin, &cErr)
	if lib.freeFunc == nil {
		err := fmt.Errorf("function free: %s", C.GoString(cErr))
		xl.Errorf("%v", err)
		return lib, err
	}

	return lib, nil
}

func (_lib Lib2) formatCreateParmas(
	ctx context.Context, params *CreateParams,
) *lib.CreateParams {
	var (
		_params = &lib.CreateParams{
			UseDevice: proto.String(params.UseDevice),
			BatchSize: proto.Int32(int32(params.BatchSize)),
			Env: &lib.CreateParams_Env{
				App:       proto.String(params.App),
				Workspace: proto.String(params.Workspace),
			},
		}
	)
	var xl = xlog.FromContextSafe(ctx)
	{
		for name, file := range params.ModelFiles {
			bs, err := ioutil.ReadFile(file)
			if err != nil {
				panic(err)
			}
			_params.ModelFiles = append(
				_params.ModelFiles,
				&lib.CreateParams_File{
					Name: proto.String(name),
					Body: bs,
				})
		}
		xl.Info("%v", params.ModelFiles)
	}
	if params.ModelParams != nil {
		bs, err := json.Marshal(params.ModelParams)
		if err != nil {
			panic(err)
		}
		_params.ModelParams = proto.String(string(bs))
	}
	{
		for name, file := range params.CustomFiles {
			bs, err := ioutil.ReadFile(file)
			if err != nil {
				panic(err)
			}
			_params.CustomFiles = append(
				_params.CustomFiles,
				&lib.CreateParams_File{
					Name: proto.String(name),
					Body: bs,
				})
		}
	}
	if params.CustomParams != nil {
		bs, err := json.Marshal(params.CustomParams)
		if err != nil {
			panic(err)
		}
		_params.CustomParams = proto.String(string(bs))
	}
	return _params
}

func (_lib *Lib2) Create(ctx context.Context, params *CreateParams) (Instance, error) {

	var (
		// xl = xlog.FromContextSafe(ctx)
		_params = _lib.formatCreateParmas(ctx, params)
	)

	maxConcurrent := params.MaxConcurrent
	if maxConcurrent == 0 {
		maxConcurrent = 24
	}
	return newLibMultiInstance2(_lib, _params, maxConcurrent)
}

var _ Instance = &LibMultiInstance2{}

type LibMultiInstance2 struct {
	*Lib2
	ctx C.PredictorContext

	instances []*LibInstance2
	ch        chan int
}

func newLibMultiInstance2(
	_lib *Lib2, params *lib.CreateParams, threads int,
) (*LibMultiInstance2, error) {

	mi := &LibMultiInstance2{
		Lib2:      _lib,
		instances: make([]*LibInstance2, 0, threads),
		ch:        make(chan int, threads),
	}

	var (
		xl = xlog.NewDummy()

		createParams, err = proto.Marshal(params)
		cErr              *C.char
	)
	if err != nil {
		return nil, err
	}

	{
		_CCreateParams := C.CBytes(createParams)
		defer C.free(_CCreateParams)
		cCode := C.callFuncCreate(mi.createFunc,
			_CCreateParams, C.int(len(createParams)), &mi.ctx)
		// if err != nil {
		// 	xl.Errorf("create failed. %v", err)
		// 	return nil, err
		// }
		if int(cCode) != 0 {
			C.callFuncGetLastError(mi.getLastErrorFunc, &cErr)
			xl.Errorf("create failed. %d %v", int(cCode), C.GoString(cErr))
			return nil, fmt.Errorf("code: %d, msg: %s", int(cCode), C.GoString(cErr))
		}
	}
	for i := 0; i < threads; i++ {
		ii, err := newLibInstance2(mi, params)
		if err != nil {
			return nil, err
		}
		mi.instances = append(mi.instances, ii)
		mi.ch <- i
	}

	return mi, nil
}

func (mi *LibMultiInstance2) Preprocess(
	_ context.Context, req model.EvalRequestInner,
) (model.EvalRequestInner, error) {
	return req, nil
}
func (mi *LibMultiInstance2) PreprocessGroup(
	_ context.Context, req model.GroupEvalRequestInner,
) (model.GroupEvalRequestInner, error) {
	return req, nil
}
func (mi *LibMultiInstance2) Inference(
	ctx context.Context, req []model.EvalRequestInner,
) ([]Response, error) {
	var index int
	select {
	case index = <-mi.ch:
	case _ = <-time.After(time.Minute):
		return nil, errors.New("overdue")
	}
	defer func() { mi.ch <- index }()
	return mi.instances[index].Inference(ctx, req)
}
func (mi *LibMultiInstance2) InferenceGroup(
	ctx context.Context, req []model.GroupEvalRequestInner,
) ([]Response, error) {
	var index int
	select {
	case index = <-mi.ch:
	case _ = <-time.After(time.Minute):
		return nil, errors.New("overdue")
	}
	defer func() { mi.ch <- index }()
	return mi.instances[index].InferenceGroup(ctx, req)
}

var _ Instance = &LibInstance2{}

type LibInstance2 struct {
	*LibMultiInstance2
	handle C.PredictorHandle

	ch chan struct {
		done chan bool
		f    func()
	}
}

func newLibInstance2(_lib *LibMultiInstance2, params *lib.CreateParams) (*LibInstance2, error) {

	i := &LibInstance2{
		LibMultiInstance2: _lib,
		ch: make(chan struct {
			done chan bool
			f    func()
		}),
	}

	var (
		xl   = xlog.NewDummy()
		wait sync.WaitGroup

		// createParams, err = proto.Marshal(params)
		createParams = []byte("") // 无用参数
		cCode        C.int
		cErr         *C.char
	)
	// if err != nil {
	// 	return nil, err
	// }

	wait.Add(1)
	go func(ctx context.Context) {
		runtime.LockOSThread()

		_CCreateParams := C.CBytes(createParams)
		// var err error
		cCode = C.callFuncHandle(i.handleFunc, i.ctx,
			_CCreateParams, C.int(len(createParams)), &i.handle)
		// if err != nil {
		// 	xl.Errorf("create handle failed. %v", err)
		// 	return // TODO
		// }
		C.free(_CCreateParams)
		if int(cCode) != 0 {
			C.callFuncGetLastError(i.getLastErrorFunc, &cErr)
			xl.Errorf("create handle failed. %d %v", int(cCode), C.GoString(cErr))
			return
		}

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

func (i *LibInstance2) Preprocess(
	ctx context.Context, req model.EvalRequestInner,
) (model.EvalRequestInner, error) {
	return req, nil
}

func (i *LibInstance2) PreprocessGroup(
	ctx context.Context, req model.GroupEvalRequestInner,
) (model.GroupEvalRequestInner, error) {
	return req, nil
}

func (i *LibInstance2) invoke(f func()) {
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

func (i *LibInstance2) inference(
	ctx context.Context, requests *lib.InferenceRequests,
) ([]Response, error) {

	var (
		xl             = xlog.FromContextSafe(ctx)
		_requests, err = proto.Marshal(requests)

		retSize C.int
		ret     unsafe.Pointer
		cCode   C.int
		cErr    *C.char
	)
	if err != nil {
		return nil, err
	}

	i.invoke(func() {
		_CRequests := C.CBytes(_requests)
		defer C.free(_CRequests)
		cCode = C.callFuncInference(i.inferenceFunc,
			i.handle,
			_CRequests, C.int(len(_requests)),
			&ret, &retSize,
		)
		if int(cCode) != 0 && int(cCode) != 200 {
			C.callFuncGetLastError(i.getLastErrorFunc, &cErr)
			xl.Errorf("inference failed. %d %v", int(cCode), C.GoString(cErr))
		}
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
		} else if str := resp.GetResult(); len(str) > 0 {
			err := json.Unmarshal([]byte(str), &_resp.Result)
			if err != nil {
				xl.Errorf("json.Unmarshal error:%v, result:%v", err, resp.GetResult())
				return nil, err
			}
		}
		responses = append(responses, _resp)
	}
	return responses, nil
}

func (i *LibInstance2) Inference(
	ctx context.Context, reqs []model.EvalRequestInner,
) ([]Response, error) {

	var (
		xl       = xlog.FromContextSafe(ctx)
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
				bs, err := ioutil.ReadFile(v.String())
				if err != nil {
					xl.Warn("%v", err)
					return nil, err
				}
				request.Data.Body = bs
			}
			if req.Data.Attribute != nil {
				bs, err := json.Marshal(req.Data.Attribute)
				if err != nil {
					xl.Warn("%v", err)
					return nil, err
				}
				request.Data.Attribute = proto.String(string(bs))
			}
		}
		if req.Params != nil {
			bs, err := json.Marshal(req.Params)
			if err != nil {
				xl.Warn("%v", err)
				return nil, err
			}
			request.Params = proto.String(string(bs))
		}
		requests.Requests = append(requests.Requests, request)
	}

	return i.inference(ctx, requests)
}

func (i *LibInstance2) InferenceGroup(
	ctx context.Context, reqs []model.GroupEvalRequestInner,
) ([]Response, error) {

	var (
		xl       = xlog.FromContextSafe(ctx)
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
					bs1, err := ioutil.ReadFile(v.String())
					if err != nil {
						xl.Warn("%v", err)
						return nil, err
					}
					data2.Body = bs1
				}
				if data.Attribute != nil {
					bs2, err := json.Marshal(data.Attribute)
					if err != nil {
						xl.Warn("%v", err)
						return nil, err
					}
					data2.Attribute = proto.String(string(bs2))
				}
				request.Datas = append(request.Datas, data2)
			}
		}
		if req.Params != nil {
			bs, err := json.Marshal(req.Params)
			if err != nil {
				xl.Warn("%v", err)
				return nil, err
			}
			request.Params = proto.String(string(bs))
		}
		requests.Requests = append(requests.Requests, request)
	}

	return i.inference(ctx, requests)
}
