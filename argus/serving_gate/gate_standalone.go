package gate

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"io"
	"io/ioutil"
	"net/http"
	"sync"
	"time"

	"github.com/qiniu/rpc.v1"
	"github.com/qiniu/rpc.v1/lb.v3"
	"github.com/qiniu/xlog.v1"
	"qiniu.com/argus/atserving/model"
	"qiniu.com/argus/com/util"
)

type gateStandalone struct {
	evals   Evals
	clients EvalClients
}

var _ Gate = &gateStandalone{}

func NewGateStandalone(evals Evals, clients EvalClients) *gateStandalone {
	return &gateStandalone{evals: evals, clients: clients}
}

func (g gateStandalone) eval(ctx context.Context, req model.TaskReq,
) (resp model.EvalResponse, rc io.ReadCloser, length int64, header http.Header, err error) {

	var (
		xl = xlog.FromContextSafe(ctx)
	)
	header = make(http.Header)

	b1 := time.Now()
	client := g.clients.GetClient(req.GetCmd(), req.GetVersion())
	if client == nil {
		xl.Errorf(
			"there is no valid client for eval with cmd:%v, version:%v",
			req.GetCmd(), req.GetVersion(),
		)
		err = errors.New("no valid hosts")
		return
	}

	rawResp, err := client.Post(ctx, req.Marshal())
	if err != nil {
		xl.Errorf("gateStandalone eval client.Post: %v", err)
		return
	}

	switch rawResp.Header.Get(model.CONTENT_TYPE) {
	case model.CT_JSON:
		model.MergeHeader(header, rawResp.Header)
		rbd, _ := ioutil.ReadAll(rawResp.Body)
		defer rawResp.Body.Close()
		if err = json.Unmarshal(rbd, &resp); err != nil {
			xl.Errorf("unmarshal result failed. %v", err)
			return
		}
	case model.CT_STREAM:
		xl.Infof(
			"eval result is stream. %s",
			rawResp.Header.Get(model.CONTENT_LENGTH),
		)
		length = rawResp.ContentLength
		rc = rawResp.Body
		model.MergeHeader(header, rawResp.Header)
	default:
		xl.Errorf("no supportted body type:%v", rawResp.Header.Get(model.CONTENT_TYPE))
		err = errors.New("the response body type is not supported")
	}

	responseTime().
		WithLabelValues(req.GetCmd(), formatVersion(req.GetVersion()), "gate.do.eval", "", "1").
		Observe(float64(time.Since(b1)) / 1e9)

	return
}

func (g gateStandalone) Eval(ctx context.Context, req model.TaskReq,
) (resp model.EvalResponse, rc io.ReadCloser, length int64, header http.Header, err error) {
	return g.eval(ctx, req)
}

func (g gateStandalone) EvalBody(
	ctx context.Context,
	cmd string, version *string,
	rc io.ReadCloser, length int64,
) (model.EvalResponse, io.ReadCloser, int64, http.Header, error) {
	var (
		xl  = xlog.FromContextSafe(ctx)
		req = model.EvalRequest{Cmd: cmd, Version: version}
	)
	xl.Infof("EvalBody cmd:%v,version:%v,length:%v", cmd, version, length)

	bd, _ := ioutil.ReadAll(rc)
	defer rc.Close()
	req.Data.URI = model.STRING(encodeDataURI(bd))
	return g.eval(ctx, req)
}

func (g gateStandalone) EvalBatch(ctx context.Context, reqs []model.TaskReq) (resps []interface{}, header http.Header, err error) {
	var (
		xl     = xlog.FromContextSafe(ctx)
		waiter sync.WaitGroup
		muter  sync.Mutex
	)

	xl.Infof("EvalBatch reqs length:%v", len(reqs))

	header = make(http.Header)
	waiter.Add(len(reqs))
	for _, req := range reqs {
		go func(ctex context.Context, r model.TaskReq) {
			defer waiter.Done()
			var uri string

			resp, rc, _, header, _err := g.eval(ctex, r)
			if _err != nil {
				err = _err
				return
			}
			if rc != nil {
				bd, _ := ioutil.ReadAll(rc)
				uri = encodeDataURI(bd)
				defer rc.Close()
			}
			muter.Lock()
			defer muter.Unlock()
			model.MergeHeader(header, header)
			if uri != "" {
				resps = append(resps, uri)
				return
			}
			resps = append(resps, resp)
		}(util.SpawnContext(ctx), req)
	}
	waiter.Wait()

	return
}

type EvalClient interface {
	Post(context.Context, []byte) (*http.Response, error)
}

type _EvalClientFunc struct {
	_Post func(context.Context, []byte) (*http.Response, error)
}

func (f _EvalClientFunc) Post(ctx context.Context, req []byte) (*http.Response, error) {
	return f._Post(ctx, req)
}

type EvalClients interface {
	GetClient(string, *string) EvalClient
	SetHosts(map[string]map[string][]string, time.Duration) // TODO
}

type evalClients struct {
	clients map[string]struct {
		Client   *lb.Client
		Versions map[string]*lb.Client
	}
	*sync.RWMutex
}

func NewEvalClients() EvalClients {
	return &evalClients{
		clients: make(map[string]struct {
			Client   *lb.Client
			Versions map[string]*lb.Client
		}),
		RWMutex: new(sync.RWMutex),
	}
}

func (s *evalClients) GetClient(cmd string, version *string) EvalClient {
	ret := func(client *lb.Client) EvalClient {

		return _EvalClientFunc{
			_Post: func(ctx context.Context, bs []byte) (resp *http.Response, err error) {
				resp, err = client.PostWith(ctx,
					"/v1/eval", "application/json",
					bytes.NewReader(bs), len(bs),
				)
				if err == nil && resp.StatusCode/100 != 2 {
					err = rpc.ResponseError(resp)
				}
				return
			},
		}
	}

	s.RLock()
	defer s.RUnlock()

	if ss, ok := s.clients[cmd]; ok {
		if version == nil {
			return ret(ss.Client)
		} else if c, ok := ss.Versions[*version]; ok {
			return ret(c)
		}
	}
	return nil
}

func (s *evalClients) SetHosts(hosts map[string]map[string][]string, timeout time.Duration) {

	var xl = xlog.NewDummy()

	if hosts == nil {
		xl.Info("nill hosts ...")
		return
	}
	if timeout == 0 {
		timeout = _DefaultEvalsTimeout
	}
	clients := make(map[string]struct {
		Client   *lb.Client
		Versions map[string]*lb.Client
	})

	for cmd, vf := range hosts {
		ss := struct {
			Client   *lb.Client
			Versions map[string]*lb.Client
		}{
			Versions: make(map[string]*lb.Client),
		}
		hosts := make([]string, 0)
		for ver, _hosts := range vf {
			hosts = append(hosts, _hosts...)
			ss.Versions[ver] = lb.New(
				&lb.Config{
					Http: &http.Client{
						Timeout: timeout,
						Transport: evalTransport{
							_EvalEnv:     _EvalEnv{Uid: 111, Utype: 1},
							RoundTripper: http.DefaultTransport,
						},
					},
					Hosts:      _hosts,
					HostRetrys: 3,
				})
		}
		ss.Client = lb.New(
			&lb.Config{
				Http: &http.Client{
					Timeout: timeout,
					Transport: evalTransport{
						_EvalEnv:     _EvalEnv{Uid: 111, Utype: 1},
						RoundTripper: http.DefaultTransport,
					},
				},
				Hosts:      hosts,
				HostRetrys: 3,
			})
		clients[cmd] = ss
	}

	s.Lock()
	defer s.Unlock()

	s.clients = clients
	xl.Infof("setHosts:%#v", s.clients)
}
