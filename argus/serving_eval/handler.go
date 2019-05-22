package eval

import (
	"context"
	"crypto/sha1"
	"encoding/hex"
	"io"
	"io/ioutil"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"sync"

	"github.com/qiniu/xlog.v1"

	"qiniu.com/argus/atserving/model"
)

const (
	_SchemeFilePrefix = "file://"
)

// Stream ...
type Stream interface {
	Name() string
	Open(context.Context) (io.ReadCloser, int64, error)
	Clean() error
}

//----------------------------------------------------------------------------//

// Handler ...
type Handler interface {
	LoadEval(context.Context, []Stream) ([]interface{}, error)
	LoadGroupEval(context.Context, [][]Stream) ([][]interface{}, error)

	PreEval(context.Context, model.EvalRequestInner) (model.EvalRequestInner, error)
	PreGroupEval(context.Context, model.GroupEvalRequestInner) (model.GroupEvalRequestInner, error)

	Eval(context.Context, []model.EvalRequestInner) ([]EvalResponseInner, error)
	GroupEval(context.Context, []model.GroupEvalRequestInner) ([]EvalResponseInner, error)
}

var _ Handler = &handler{}

type handler struct {
	Core
	dir string
}

// NewHandler ...
func NewHandler(core Core, dir string) Handler {
	return &handler{
		Core: core,
		dir:  dir,
	}
}

func (h *handler) filename(name ...string) string {
	sum := sha1.Sum([]byte(strings.Join(name, "_")))
	return filepath.Join(h.dir, hex.EncodeToString(sum[:]))
}

func (h *handler) copy2File(ctx context.Context, s Stream, key ...string) (filename string, err error) {
	filename = h.filename(append(key, s.Name())...)
	file, err := os.OpenFile(filename, os.O_RDWR|os.O_CREATE, 0755)
	if err != nil {
		return
	}
	defer file.Close()
	filename = _SchemeFilePrefix + filename
	rc, size, err := s.Open(ctx)
	if err != nil {
		return
	}
	defer rc.Close()
	if size <= 0 {
		_, err = io.Copy(file, rc)
	} else {
		_, err = io.CopyN(file, rc, size)
	}
	return
}

func (h *handler) copy2Mem(ctx context.Context, s Stream) (bs model.BYTES, err error) {
	rc, _, err := s.Open(ctx)
	if err != nil {
		return
	}
	defer rc.Close()
	bs, err = ioutil.ReadAll(rc)
	return
}

func (h *handler) LoadEval(ctx context.Context, streams []Stream) ([]interface{}, error) {

	var uris = make([]interface{}, len(streams))
	xl := xlog.FromContextSafe(ctx)
	pool := newCopyFilePool(5) // TODO config
	for i, stream := range streams {
		var (
			key = strconv.Itoa(i)
			s   = stream
		)
		ctx2 := xlog.NewContext(ctx, xl.Spawn())
		pool.Add(func() (context.Context, string, interface{}, error) {
			bs, err := h.copy2Mem(ctx2, s)
			return ctx2, key, bs, err
		})
	}
	result := pool.Wait()
	for i := range streams {
		var (
			key  = strconv.Itoa(i)
			rest = result[key]
		)
		xl.Xput(xlog.FromContextSafe(rest.Context).Xget())
		if rest.Error != nil {
			return nil, rest.Error
		}
		uris[i] = rest.Ret
	}
	return uris, nil
}

func (h *handler) LoadGroupEval(ctx context.Context, streams [][]Stream) ([][]interface{}, error) {

	var uris = make([][]interface{}, len(streams))
	xl := xlog.FromContextSafe(ctx)
	pool := newCopyFilePool(5) // TODO config
	for i, ss := range streams {
		for j, stream := range ss {
			var key2 = strconv.Itoa(i) + "_" + strconv.Itoa(j)
			var s = stream
			ctx2 := xlog.NewContext(ctx, xl.Spawn())
			pool.Add(func() (context.Context, string, interface{}, error) {
				bs, err := h.copy2Mem(ctx2, s)
				return ctx2, key2, bs, err
			})
		}
	}
	rests := pool.Wait()
	for i, ss := range streams {
		rs := make([]interface{}, 0, len(ss))
		for j := range ss {
			var key2 = strconv.Itoa(i) + "_" + strconv.Itoa(j)
			var rest = rests[key2]
			xl.Xput(xlog.FromContextSafe(rest.Context).Xget())
			if rest.Error != nil {
				return nil, rest.Error
			}
			rs = append(rs, rest.Ret)
		}
		uris[i] = rs
	}
	return uris, nil
}

func (h *handler) PreEval(ctx context.Context, req model.EvalRequestInner) (
	model.EvalRequestInner, error) {

	switch v := req.Data.URI.(type) {
	case model.STRING:
		req.Data.URI = model.STRING(strings.TrimPrefix(v.String(), _SchemeFilePrefix))
	}
	req, err := h.Core.PreEval(ctx, req)
	if err != nil {
		return req, err
	}
	switch v := req.Data.URI.(type) {
	case model.STRING:
		req.Data.URI = model.STRING(_SchemeFilePrefix + v.String())
	}
	return req, err
}

func (h *handler) PreGroupEval(
	ctx context.Context,
	req model.GroupEvalRequestInner,
) (model.GroupEvalRequestInner, error) {
	for i, data := range req.Data {
		switch v := data.URI.(type) {
		case model.STRING:
			req.Data[i].URI = model.STRING(strings.TrimPrefix(v.String(), _SchemeFilePrefix))
		}
	}
	req, err := h.Core.PreGroupEval(ctx, req)
	if err != nil {
		return req, err
	}
	for i, data := range req.Data {
		switch v := data.URI.(type) {
		case model.STRING:
			req.Data[i].URI = model.STRING(_SchemeFilePrefix + v.String())
		}
	}
	return req, err
}

func (h *handler) Eval(
	ctx context.Context,
	reqs []model.EvalRequestInner,
) (resps []EvalResponseInner, err error) {

	for i := range reqs {
		switch v := reqs[i].Data.URI.(type) {
		case model.STRING:
			reqs[i].Data.URI = model.STRING(strings.TrimPrefix(v.String(), _SchemeFilePrefix))
		}
	}

	resps, err = h.Core.Eval(ctx, reqs)
	if err != nil {
		return nil, err
	}

	return resps, nil
}

func (h *handler) GroupEval(
	ctx context.Context,
	reqs []model.GroupEvalRequestInner,
) (resps []EvalResponseInner, err error) {

	for i := range reqs {
		for j := range reqs[i].Data {
			switch v := reqs[i].Data[j].URI.(type) {
			case model.STRING:
				reqs[i].Data[j].URI = model.STRING(strings.TrimPrefix(v.String(), _SchemeFilePrefix))
			}
		}
	}

	resps, err = h.Core.GroupEval(ctx, reqs)
	if err != nil {
		return nil, err
	}

	return resps, nil
}

////////////////////////////////////////////////////////////////////////////////

type copyFileResult struct {
	context.Context
	Ret   interface{}
	Error error
}

// CopyFilePool ...
type CopyFilePool interface {
	Add(func() (context.Context, string, interface{}, error))
	Wait() map[string]copyFileResult
	Err() error
}

type copyFilePool struct {
	ch     chan func() (context.Context, string, interface{}, error)
	result map[string]copyFileResult
	group  sync.WaitGroup
	err    error
	*sync.Mutex
}

func newCopyFilePool(n int) CopyFilePool {
	pool := &copyFilePool{
		ch:     make(chan func() (context.Context, string, interface{}, error)),
		result: make(map[string]copyFileResult),
		Mutex:  new(sync.Mutex),
	}
	for i := 0; i < n; i++ {
		pool.group.Add(1)
		go func() {
			for f := range pool.ch {
				func() {
					defer func() {
						if r := recover(); r != nil {
							err := recoverAsError(r)
							pool.Lock()
							defer pool.Unlock()
							if pool.err == nil {
								pool.err = err
							}
						}
					}()
					ctx, key, ret, err := f()
					pool.Lock()
					defer pool.Unlock()
					pool.result[key] = copyFileResult{Context: ctx, Ret: ret, Error: err}
				}()
			}
			pool.group.Done()
		}()
	}
	return pool
}

func (pool *copyFilePool) Add(f func() (context.Context, string, interface{}, error)) {
	pool.ch <- f
}

func (pool *copyFilePool) Wait() map[string]copyFileResult {
	close(pool.ch)
	pool.group.Wait()
	return pool.result
}

func (pool *copyFilePool) Err() error { return pool.err }
