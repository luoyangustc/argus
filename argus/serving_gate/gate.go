package gate

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"io/ioutil"
	"net/http"
	"strconv"
	"sync"
	"time"

	"github.com/qiniu/http/httputil.v1"
	"github.com/qiniu/xlog.v1"

	"qiniu.com/argus/atserving/model"
	STS "qiniu.com/argus/sts/client"
)

// Gate ...
type Gate interface {
	Eval(context.Context, model.TaskReq,
	) (model.EvalResponse, io.ReadCloser, int64, http.Header, error)
	EvalBody(
		ctx context.Context,
		cmd string, version *string,
		rc io.ReadCloser, length int64,
	) (model.EvalResponse, io.ReadCloser, int64, http.Header, error)

	// EvalResponse OR Data URI
	EvalBatch(context.Context, []model.TaskReq) ([]interface{}, http.Header, error)
}

//----------------------------------------------------------------------------//

// NewGate ...
func NewGate(p Producer, w Worker, sts STS.Client, e Evals) Gate {
	return &gate{
		Producer: p,
		Client:   sts,
		Worker:   w,
		Evals:    e,
	}
}

var _ Gate = &gate{}

type gate struct {
	Producer
	STS.Client
	Worker
	Evals
}

func (g *gate) postEval(ctx context.Context, v *model.EvalRequest) (err error) {
	xl := xlog.FromContextSafe(ctx)
	if isBadURI(v.Data.URI.String()) {
		return ErrBadURI
	}
	if isDataURI(v.Data.URI.String()) {
		var bs []byte
		bs, err = decodeDataURI(v.Data.URI.String())
		if err != nil {
			xl.Warnf("decode data uri failed. %v", err)
			err = ErrBadDataURI
			return
		}
		var n = int64(len(bs))
		var _uri string
		_uri, err = g.NewURL(ctx, &n)
		if err != nil {
			xl.Errorf("new url failed. %v", err)
			return
		}
		v.Data.URI = model.STRING(_uri)
		wg := new(sync.WaitGroup)
		wg.Add(1)
		done := func(err1 error) {
			err = err1
			wg.Done()
		}
		go func(xl *xlog.Logger) {
			if err := g.SyncPost(
				xlog.NewContext(ctx, xl),
				v.Data.URI.String(),
				n, bytes.NewBuffer(bs),
				done,
			); err != nil {
				xl.Errorf("post data uri failed. %s %d %v", v.Data.URI, n, err)
			}
		}(xl.Spawn())
		wg.Wait()
	} else if isStsURI(v.Data.URI.String()) {
		// NOTHING TO DO
	} else {
		var _uri string
		if _uri, _, err = g.DoFetch(ctx, v.Data.URI.String(), nil, true); err != nil {
			xl.Errorf("do fetch failed. %s %s", v.Data.URI, err)
			return
		}
		v.Data.URI = model.STRING(_uri)
	}
	return
}

func (g *gate) postGroupEval(ctx context.Context, v *model.GroupEvalRequest) (err error) {
	xl := xlog.FromContextSafe(ctx)
	wg := new(sync.WaitGroup)
	for i := range v.Data {
		if isBadURI(v.Data[i].URI.String()) {
			return ErrBadURI
		}
		if isDataURI(v.Data[i].URI.String()) {
			var bs []byte
			bs, err = decodeDataURI(v.Data[i].URI.String())
			if err != nil {
				xl.Warnf("decode data uri failed. %v", err)
				err = ErrBadDataURI
				return
			}
			var n = int64(len(bs))
			var _uri string
			_uri, err = g.NewURL(ctx, &n)
			if err != nil {
				xl.Errorf("new url failed. %v", err)
				return
			}
			v.Data[i].URI = model.STRING(_uri)
			wg.Add(1)
			done := func(err error) {
				wg.Done()
				// TODO 处理err
			}
			go func(index int, xl *xlog.Logger) {
				if err := g.SyncPost(
					xlog.NewContext(ctx, xl),
					v.Data[index].URI.String(), n, bytes.NewBuffer(bs),
					done,
				); err != nil {
					xl.Errorf("post data uri failed. %s %d %v", v.Data[index].URI, n, err)
				}
			}(i, xl.Spawn())
		} else if isStsURI(v.Data[i].URI.String()) {
			// NOTHING TO DO
		} else {
			var _uri string
			if _uri, _, err = g.DoFetch(ctx, v.Data[i].URI.String(), nil, true); err != nil {
				xl.Errorf("do fetch failed. %s %s", v.Data[i].URI, err)
				return
			}
			v.Data[i].URI = model.STRING(_uri)
		}
	}
	wg.Wait()
	return
}

func (g *gate) eval(
	ctx context.Context,
	req model.TaskReq,
) (resp model.EvalResponse, rc io.ReadCloser, length int64, header http.Header, err error) {

	var (
		xl       = xlog.FromContextSafe(ctx)
		resps    []*model.ResponseMessage
		respM    *model.ResponseMessage
		duration = g.Timeout(req.GetCmd(), req.GetVersion())
	)
	header = make(http.Header)

	b1 := time.Now()

	defer func() {
		var errCode = 200
		if err != nil {
			errCode = httputil.DetectCode(err)
		}
		requestsCounter(req.GetCmd(), "gate.do", strconv.Itoa(errCode)).Inc()
	}()

	if resps, err = g.Do(ctx, duration, []model.TaskReq{req}, g.Publish); err != nil {
		xl.Errorf("worker do failed. %s", err)
		return
	}
	respM = resps[0]

	responseTime().
		WithLabelValues(req.GetCmd(), formatVersion(req.GetVersion()), "gate.do", "", "").
		Observe(float64(time.Since(b1)) / 1e9)

	if respM.StatusCode != http.StatusOK {
		err = httputil.NewError(respM.StatusCode, respM.StatusText)
		return
	}

	switch respM.Header.Get(model.CONTENT_TYPE) {
	case model.CT_JSON:
		model.MergeHeader(header, respM.Header)
		if err = json.Unmarshal([]byte(respM.Response), &resp); err != nil {
			xl.Errorf("unmarshal result failed. %v", err)
			return
		}
	case model.CT_STREAM:
		xl.Infof(
			"eval result is stream. %s %s",
			respM.Response, respM.Header.Get(model.CONTENT_LENGTH),
		)
		length, _ = strconv.ParseInt(respM.Header.Get(model.CONTENT_LENGTH), 10, 64)
		var uri string
		if err = json.Unmarshal([]byte(respM.Response), &uri); err != nil {
			xl.Errorf("unmarshal result failed. %v", err)
		}
		model.MergeHeader(header, respM.Header)
		rc, _, _, err = g.Get(ctx, uri, &length)
	}

	return
}

func (g *gate) Eval(
	ctx context.Context,
	req model.TaskReq,
) (resp model.EvalResponse, rc io.ReadCloser, length int64, header http.Header, err error) {
	reqProcessing.Inc()
	defer reqProcessing.Dec()

	b1 := time.Now()
	switch req.(type) {
	case model.EvalRequest:
		v := req.(model.EvalRequest)
		if err = g.postEval(ctx, &v); err != nil {
			return
		}
		req = v
	case model.GroupEvalRequest:
		v := req.(model.GroupEvalRequest)
		if err = g.postGroupEval(ctx, &v); err != nil {
			return
		}
		req = v
	}
	responseTime().
		WithLabelValues(req.GetCmd(), formatVersion(req.GetVersion()), "gate.doFetch", "", "1").
		Observe(float64(time.Since(b1)) / 1e9)

	return g.eval(ctx, req)
}

func (g *gate) EvalBatch(
	ctx context.Context, reqs []model.TaskReq,
) (resps []interface{}, header http.Header, err error) {
	numReq := float64(len(reqs))
	reqProcessing.Add(numReq)
	defer reqProcessing.Sub(numReq)

	var (
		xl       = xlog.FromContextSafe(ctx)
		respMs   []*model.ResponseMessage
		duration time.Duration
	)
	header = make(http.Header)

	b1 := time.Now()
	for i, req := range reqs {
		if d := g.Timeout(req.GetCmd(), req.GetVersion()); d > duration {
			duration = d
		}
		switch req.(type) {
		case model.EvalRequest:
			v := req.(model.EvalRequest)
			if err = g.postEval(ctx, &v); err != nil {
				return
			}
			reqs[i] = v
		case model.GroupEvalRequest:
			v := req.(model.GroupEvalRequest)
			if err = g.postGroupEval(ctx, &v); err != nil {
				return
			}
			reqs[i] = v
		}
	}

	responseTime().
		WithLabelValues("batch", formatVersion(nil), "gate.doFetch", "", strconv.Itoa(len(reqs))).
		Observe(float64(time.Since(b1)) / 1e9)

	b2 := time.Now()
	if respMs, err = g.Do(ctx, duration, reqs, g.Publish); err != nil {
		xl.Errorf("worker do failed. %s", err)
		return
	}

	responseTime().
		WithLabelValues("batch", formatVersion(nil), "gate.do", "", "").
		Observe(float64(time.Since(b2)) / 1e9)

	for _, respM := range respMs {
		model.MergeHeader(header, respM.Header)
		if respM.StatusCode != http.StatusOK {
			err = httputil.NewError(respM.StatusCode, respM.StatusText)
			return
		}
		switch respM.Header.Get(model.CONTENT_TYPE) {
		case model.CT_JSON:
			var resp model.EvalResponse
			if err = json.Unmarshal([]byte(respM.Response), &resp); err != nil {
				xl.Errorf("unmarshal result failed. %s", err)
				return
			}
			resps = append(resps, resp)
		case model.CT_STREAM:
			var (
				length int64
				uri    string
				rc     io.ReadCloser
			)
			xl.Infof(
				"eval result is stream. %s %s",
				respM.Response, respM.Header.Get(model.CONTENT_LENGTH),
			)
			length, _ = strconv.ParseInt(respM.Header.Get(model.CONTENT_LENGTH), 10, 64)
			if err = json.Unmarshal([]byte(respM.Response), &uri); err != nil {
				xl.Errorf("unmarshal result failed. %v", err)
				return
			}
			uri, err = func() (string, error) {
				rc, _, _, err = g.Get(ctx, uri, &length)
				if err != nil {
					return "", err
				}
				defer rc.Close()
				bs, err := ioutil.ReadAll(rc)
				if err != nil {
					return "", err
				}
				return encodeDataURI(bs), nil
			}()
			if err != nil {
				xl.Errorf("unmarshal result failed. %v", err)
				return
			}
			resps = append(resps, model.STRING(uri))
		}
	}

	return
}

func (g *gate) EvalBody(
	ctx context.Context,
	cmd string, version *string,
	rc1 io.ReadCloser, length int64,
) (resp model.EvalResponse, rc2 io.ReadCloser, length2 int64, header http.Header, err error) {
	reqProcessing.Inc()
	defer reqProcessing.Dec()

	var (
		xl  = xlog.FromContextSafe(ctx)
		req = model.EvalRequest{Cmd: cmd, Version: version}
	)
	_uri, _ := g.NewURL(ctx, &length)
	req.Data.URI = model.STRING(_uri)
	defer rc1.Close()
	if err = g.Post(ctx, req.Data.URI.String(), length, rc1); err != nil {
		xl.Errorf("post failed. %s %d %s", req.Data.URI, length, err)
		return
	}
	return g.eval(ctx, req)
}

//----------------------------------------------------------------------------//

// Evals ...
type Evals interface {
	IsAllowable(uint32, string) bool
	Available(string, *string) bool
	Timeout(string, *string) time.Duration

	SetWorkerDefault(model.ConfigWorker)
	SetWorker(string, *string, model.ConfigWorker)
	UnsetWorker(string, *string)

	SetAppMetadataDefault(model.ConfigAppMetadata)
	SetAppMetadata(string, model.ConfigAppMetadata)
	UnsetAppMetadata(string)

	Register(string, string, model.ConfigAppRelease)
	Unregister(string, string)
}

var (
	_DefaultEvalsTimeout = time.Second * 10
)

var _ Evals = &evals{}

type evals struct {
	workerDefault       model.ConfigWorker
	workerOfApps        map[string]model.ConfigWorker
	workerOfAppReleases map[string]map[string]model.ConfigWorker

	appMetadataDefault model.ConfigAppMetadata
	appMetadatas       map[string]model.ConfigAppMetadata

	appReleases map[string]map[string]model.ConfigAppRelease

	*sync.RWMutex
}

// NewEvals ...
func NewEvals() Evals {
	e := &evals{
		workerDefault:       model.ConfigWorker{Timeout: _DefaultEvalsTimeout},
		workerOfApps:        make(map[string]model.ConfigWorker),
		workerOfAppReleases: make(map[string]map[string]model.ConfigWorker),
		appMetadataDefault:  model.ConfigAppMetadata{},
		appMetadatas:        make(map[string]model.ConfigAppMetadata),
		appReleases:         make(map[string]map[string]model.ConfigAppRelease),
		RWMutex:             new(sync.RWMutex),
	}
	return e
}

func (e *evals) IsAllowable(uid uint32, cmd string) bool {
	e.RLock()
	defer e.RUnlock()

	if meta, ok := e.appMetadatas[cmd]; ok {
		if meta.Public {
			return true
		}
		if uid == meta.Owner.UID {
			return true
		}
		if meta.UserWhiteList != nil && len(meta.UserWhiteList) > 0 {
			for _, _uid := range meta.UserWhiteList {
				if _uid == uid {
					return true
				}
			}
			return false
		}
	}
	if e.appMetadataDefault.Public { // 临时代码，保证CS环境可随意测试
		return true
	}
	for _, _uid := range e.appMetadataDefault.UserWhiteList {
		if _uid == uid {
			return true
		}
	}
	return false
}

func (e *evals) Available(cmd string, version *string) bool {
	e.RLock()
	defer e.RUnlock()

	m, ok := e.appReleases[cmd]
	if !ok {
		return false
	}

	if version != nil {
		_, ok := m[*version]
		return ok
	}

	return true
}

func (e *evals) Timeout(cmd string, version *string) time.Duration {
	e.RLock()
	defer e.RUnlock()
	if version != nil && e.workerOfAppReleases != nil {
		if m, ok := e.workerOfAppReleases[cmd]; ok {
			if c, ok := m[*version]; ok {
				return c.Timeout
			}
		}
	}
	if e.workerOfApps != nil {
		if c, ok := e.workerOfApps[cmd]; ok {
			return c.Timeout
		}
	}
	return e.workerDefault.Timeout
}

func (e *evals) SetWorkerDefault(conf model.ConfigWorker) {
	e.Lock()
	defer e.Unlock()

	e.workerDefault = conf
}
func (e *evals) SetWorker(cmd string, version *string, conf model.ConfigWorker) {
	e.Lock()
	defer e.Unlock()

	if version == nil {
		e.workerOfApps[cmd] = conf
	} else {
		m, ok := e.workerOfAppReleases[cmd]
		if !ok {
			m = make(map[string]model.ConfigWorker)
		}
		m[*version] = conf
		e.workerOfAppReleases[cmd] = m
	}
}
func (e *evals) UnsetWorker(cmd string, version *string) {
	e.Lock()
	defer e.Unlock()

	if version == nil {
		delete(e.workerOfApps, cmd)
	} else {
		m, ok := e.workerOfAppReleases[cmd]
		if ok {
			delete(m, *version)
			if len(m) > 0 {
				e.workerOfAppReleases[cmd] = m
			} else {
				delete(e.workerOfAppReleases, cmd)
			}
		}
	}
}
func (e *evals) SetAppMetadataDefault(meta model.ConfigAppMetadata) {
	e.Lock()
	defer e.Unlock()

	e.appMetadataDefault = meta
}
func (e *evals) SetAppMetadata(cmd string, meta model.ConfigAppMetadata) {
	e.Lock()
	defer e.Unlock()

	e.appMetadatas[cmd] = meta
}
func (e *evals) UnsetAppMetadata(cmd string) {
	e.Lock()
	defer e.Unlock()

	delete(e.appMetadatas, cmd)
}
func (e *evals) Register(cmd, version string, conf model.ConfigAppRelease) {
	e.Lock()
	defer e.Unlock()

	m, ok := e.appReleases[cmd]
	if !ok {
		m = make(map[string]model.ConfigAppRelease)
	}
	m[version] = conf
	e.appReleases[cmd] = m
}
func (e *evals) Unregister(cmd, version string) {
	e.Lock()
	defer e.Unlock()

	m, ok := e.appReleases[cmd]
	if ok {
		delete(m, version)
		if len(m) > 0 {
			e.appReleases[cmd] = m
		} else {
			delete(e.appReleases, cmd)
		}
	}
}
