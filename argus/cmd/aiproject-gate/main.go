package main

import (
	"crypto/tls"
	"crypto/x509"
	"io/ioutil"
	"net/http"
	"os"
	"reflect"
	"runtime"
	"strings"

	"github.com/qbox/ke-base/sdk"
	"github.com/qiniu/errors"
	"github.com/qiniu/http/restrpc.v1"
	"github.com/qiniu/http/rpcutil.v1"
	"github.com/qiniu/http/servestk.v1"
	"github.com/qiniu/log.v1"
	"qbox.us/cc/config"
	jsonlog "qbox.us/http/audit/jsonlog.v3"
	aiprjgate "qiniu.com/argus/argus/aiproject/gate"
)

var (
	PORT_HTTP string
	APP_PORT  string
	CA_PEM    string
	CA_KEY    string
)

func init() {
	PORT_HTTP = os.Getenv("PORT_HTTP")
	APP_PORT = os.Getenv("APP_PORT")
	CA_PEM = os.Getenv("CA_PEM")
	CA_KEY = os.Getenv("CA_KEY")
}

type Config struct {
	Version      string                 `json:"version"`
	HTTPHost     string                 `json:"http_host"`
	CaPem        string                 `json:"ca_pem"`
	CaKey        string                 `json:"ca_key"`
	AuditLog     jsonlog.Config         `json:"audit_log"`
	RouterConfig aiprjgate.RouterConfig `json:"router_configure"`
	DebugLevel   int                    `json:"debug_level"`
}

func main() {

	config.Init("f", "aiproject", "aiproject.conf")
	var conf Config
	if err := config.Load(&conf); err != nil {
		log.Fatal("Failed to load configure file!")
	}

	log.SetOutputLevel(conf.DebugLevel)
	log.Debugf("load conf %#v", conf)
	runtime.GOMAXPROCS(runtime.NumCPU())

	if strings.TrimSpace(PORT_HTTP) != "" {
		conf.HTTPHost = "0.0.0.0:" + PORT_HTTP
	}
	if strings.TrimSpace(APP_PORT) != "" {
		conf.RouterConfig.AppsConf.Port = APP_PORT
	}
	if strings.TrimSpace(CA_PEM) != "" {
		conf.CaPem = CA_PEM
	}
	if strings.TrimSpace(CA_KEY) != "" {
		conf.CaKey = CA_KEY
	}

	tlsConf, err := TLSConfig(conf.CaPem, conf.CaKey)
	if err != nil {
		log.Fatal("load tls key pair error:", errors.Detail(err))
	}

	microCl := sdk.New(sdk.Config{
		Username: conf.RouterConfig.K8sConf.Username,
		Password: conf.RouterConfig.K8sConf.Password,
		Host:     conf.RouterConfig.K8sConf.Host,
	}).MicroService(conf.RouterConfig.K8sConf.Region)
	reRouter := aiprjgate.NewRouter(conf.RouterConfig, microCl)
	srv, err := aiprjgate.New(reRouter)
	if err != nil {
		log.Fatal("argus.New:", errors.Detail(err))
	}

	method, _ := reflect.TypeOf(srv).MethodByName("Do")
	handler, _ := rpcutil.HandlerCreator{}.New(reflect.ValueOf(srv), method)

	al, logf, err := jsonlog.Open("ARGUS-FACEC", &conf.AuditLog, nil)
	if err != nil {
		log.Fatal("jsonlog.Open failed:", errors.Detail(err))
	}
	defer logf.Close()

	alMux := servestk.New(restrpc.NewServeMux(), al.Handler)
	alMux.HandleFunc("GET /health", func(w http.ResponseWriter, r *http.Request) {
		w.Write([]byte("ok argus-facex"))
	})
	mux := http.NewServeMux()
	alMux.SetDefault(mux)

	mux.Handle("/", handler)

	hsrv := http.Server{
		Addr:      conf.HTTPHost,
		TLSConfig: tlsConf,
		Handler:   alMux,
	}
	if err := hsrv.ListenAndServeTLS("", ""); err != nil {
		log.Errorf("aiproject-gate start error: %v", err)
	}

}

func TLSConfig(caPem, caKey string) (*tls.Config, error) {

	cert, err := tls.LoadX509KeyPair(caPem, caKey)
	if err != nil {
		return nil, err
	}
	caP, _ := ioutil.ReadFile(caPem)

	pool := x509.NewCertPool()
	pool.AppendCertsFromPEM(caP)
	tconf := tls.Config{
		Certificates: []tls.Certificate{cert}, //给client校验server, 可以不要
		ClientAuth:   tls.RequireAndVerifyClientCert,
		ClientCAs:    pool,
	}

	return &tconf, nil
}
