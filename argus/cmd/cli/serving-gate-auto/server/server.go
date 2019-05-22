package server

import (
	"bytes"
	"context"
	"io"
	"io/ioutil"
	"log"
	"net/http"
	"sync"
	"time"

	"github.com/qiniu/xlog.v1"
	"qiniu.com/auth/qiniumac.v1"
)

type Config struct {
	Ak      string `json:"ak"`
	Sk      string `json:"sk"`
	Addr    string `json:"addr"`
	StsHost string `json:"sts_host"`
}

type server struct {
	cfg        Config
	lock       sync.RWMutex
	serviceMap map[string]*ConsulService
	mock       bool
}

func New(cfg Config) *server {
	return &server{cfg: cfg}
}

func (s *server) hand(w http.ResponseWriter, r *http.Request) {
	if r.URL.Path == "/" {
		html := s.appStatusPage()
		w.Header().Set("Content-Type", "text/html")
		w.Write([]byte(html))
		return
	}
	xl := xlog.New(w, r)
	app := getAppName(r.URL.Path)
	var bodyBufToLog []byte
	bodyStream := io.LimitReader(r.Body, MaxBodySize)
	bodyBuf, err := ioutil.ReadAll(bodyStream)
	if err != nil {
		xl.Warn("read req ", err)
	}
	if len(bodyBuf) < 1024 {
		bodyBufToLog = bodyBuf
	}

	r.Header.Del("Host")
	r.Header.Set("User-Agent", "SERVING_GATE_AUTO")

	proxyToCs := func() {
		req, err := http.NewRequest("POST", "http://ava-serving-gate.cs.cg.dora-internal.qiniu.io:5001"+r.URL.Path, bytes.NewReader(bodyBuf))
		req.Header = r.Header
		ce(err)
		resp, err := http.DefaultClient.Do(req)
		if err != nil {
			xl.Error("post error", err)
			w.WriteHeader(500)
			w.Write([]byte(err.Error()))
		}
		defer resp.Body.Close()
		copyHeader(w.Header(), resp.Header)
		w.Header().Set("X-SERVING-GATE-AUTO-ENV", "CS")
		w.WriteHeader(resp.StatusCode)
		if resp.StatusCode > 299 {
			xl.Warn("status", resp.StatusCode)
		}
		_, _ = io.Copy(w, resp.Body)
	}
	proxyToPrd := func() {
		req, err := http.NewRequest("POST", "http://serve.atlab.ai"+r.URL.Path, bytes.NewReader(s.fixBodyBuf(bodyBuf)))
		req.Header = r.Header
		ce(err)
		client := http.Client{
			Transport: qiniumac.NewTransport(&qiniumac.Mac{AccessKey: s.cfg.Ak, SecretKey: []byte(s.cfg.Sk)}, http.DefaultTransport),
		}
		resp, err := client.Do(req)
		if err != nil {
			xl.Error("post error", err)
			w.WriteHeader(500)
			w.Write([]byte(err.Error()))
		}
		defer resp.Body.Close()
		copyHeader(w.Header(), resp.Header)
		w.Header().Set("X-SERVING-GATE-AUTO-ENV", "PRD")
		w.WriteHeader(resp.StatusCode)
		if resp.StatusCode > 299 {
			xl.Warn("status", resp.StatusCode)
		}
		_, _ = io.Copy(w, resp.Body)
	}

	exists := s.appExists(app)

	action := "proxyToPrd"
	do := proxyToPrd
	if exists && r.Header.Get("X-USE-ENV") != "PRD" {
		do = proxyToCs
		action = "proxyToCs"
	}
	xl.Debugf("receive path:%v, app:%v, action:%v, body:%v", r.URL.Path, app, action, string(bodyBufToLog))
	do()
}

func (s *server) Run(ctx context.Context) {
	s.serviceMap = make(map[string]*ConsulService)
	xlog.SetOutputLevel(0)
	log.SetFlags(log.LstdFlags | log.Lshortfile)
	s.readAppStatus()
	go func() {
		for {
			select {
			case <-ctx.Done():
				return
			default:
			}
			s.readAppStatus()
			time.Sleep(time.Second * 10)
		}
	}()
	xl.Info("run on ", s.cfg.Addr)
	ce(http.ListenAndServe(s.cfg.Addr, http.HandlerFunc(s.hand)))
}
