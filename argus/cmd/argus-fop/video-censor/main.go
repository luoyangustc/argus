package main

import (
	"context"
	"net"
	"net/http"
	"os"
	"runtime"
	"strings"

	"github.com/qiniu/errors"
	"github.com/qiniu/http/restrpc.v1"
	"github.com/qiniu/http/servestk.v1"
	"github.com/qiniu/log.v1"
	"github.com/qiniu/xlog.v1"

	"qbox.us/cc/config"
	jsonlog "qbox.us/http/audit/jsonlog.v3"
	"qiniu.com/argus/argus/monitor"
	"qiniu.com/argus/com/proxy/fop"
	fop_proxy "qiniu.com/argus/fop/proxy"
)

type Config struct {
	HTTPHost   string         `json:"http_host"`
	AuditLog   jsonlog.Config `json:"audit_log"`
	DebugLevel int            `json:"debug_level"`
}

var (
	PORT_HTTP string
	ARGUS_URL string
)

func init() {
	PORT_HTTP = os.Getenv("PORT_HTTP")
	ARGUS_URL = os.Getenv("ARGUS_URL")
}

func main() {

	runtime.GOMAXPROCS(runtime.NumCPU())

	config.Init("f", "argus-fop", "argus-fop.conf")
	var conf Config
	if err := config.Load(&conf); err != nil {
		log.Fatal("Failed to load configure file!")
	}
	var host string
	if strings.TrimSpace(PORT_HTTP) != "" {
		conf.HTTPHost = "0.0.0.0:" + PORT_HTTP
		host = net.JoinHostPort(localHost(context.Background()), PORT_HTTP)
	} else {
		host = conf.HTTPHost
	}

	log.SetOutputLevel(conf.DebugLevel)
	log.Debugf("load conf %#v", conf)

	if ARGUS_URL == "" {
		log.Fatalln("bad argus url.")
	}

	var proxy fop.Proxy = fop_proxy.NewVideoCensor(ARGUS_URL)

	service := fop.NewService("http://"+host, proxy)

	al, logf, err := jsonlog.Open("ARGUS-FOP", &conf.AuditLog, nil)
	if err != nil {
		log.Fatal("jsonlog.Open failed:", errors.Detail(err))
	}
	defer logf.Close()

	alMux := servestk.New(restrpc.NewServeMux(), al.Handler)
	alMux.HandleFunc("GET /health", func(w http.ResponseWriter, r *http.Request) {
		w.Write([]byte("ok argus-fop"))
	})
	alMux.Handle("GET /metrics", monitor.Handler())

	router := restrpc.Router{
		PatternPrefix: "",
		Mux:           alMux,
	}

	if err := router.ListenAndServe(conf.HTTPHost, service); err != nil {
		log.Errorf("argus start error: %v", err)
	}
}

func localHost(ctx context.Context) string {
	xl := xlog.FromContextSafe(ctx)
	addrs, err := net.InterfaceAddrs()
	if err != nil {
		xl.Fatalf("get net addrs failed. %v", err)
	}
	for _, addr := range addrs {
		if host, ok := isAddrLocal(addr); ok {
			return host
		}
	}
	return ""
}

func isAddrLocal(addr net.Addr) (string, bool) {
	if ipnet, ok := addr.(*net.IPNet); !ok {
		return "", false
	} else if ipnet.IP.To4() == nil {
		return "", false
	} else {
		ip := ipnet.IP.To4()
		if ip[0] == 10 {
			return ip.String(), true
		} else if ip[0] == 172 && ip[1] >= 16 && ip[1] <= 31 {
			return ip.String(), true
		} else if ip[0] == 192 && ip[1] == 168 {
			return ip.String(), true
		}
	}
	return "", false
}
