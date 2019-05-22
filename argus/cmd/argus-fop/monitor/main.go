package main

import (
	"context"
	"net/http"
	"os"
	"runtime"
	"sync"
	"time"

	"github.com/prometheus/client_golang/prometheus/promhttp"

	"github.com/qiniu/http/restrpc.v1"
	"github.com/qiniu/http/servestk.v1"
	"github.com/qiniu/log.v1"

	"qiniu.com/argus/fop/monitor"
)

var (
	PORT_HTTP string
)

func init() {
	PORT_HTTP = os.Getenv("PORT_HTTP")
}

func main() {

	runtime.GOMAXPROCS(runtime.NumCPU())

	var host string = "0.0.0.0:" + PORT_HTTP

	mux := servestk.New(restrpc.NewServeMux())
	mux.HandleFunc("GET /health", func(w http.ResponseWriter, r *http.Request) {
		w.Write([]byte("ok argus-monitor"))
	})
	mux.Handle("GET /metrics", promhttp.Handler())

	run()

	if err := http.ListenAndServe(host, mux); err != nil {
		log.Errorf("argus start error: %v", err)
	}
}

func run() {

	go func() {
		for _ = range time.Tick(2 * time.Second) {
			var wg sync.WaitGroup
			wg.Add(1)
			go func() {
				defer wg.Done()
				monitor.MonitorPulp(context.Background())
			}()
			wg.Add(1)
			go func() {
				defer wg.Done()
				monitor.MonitorCensor(context.Background())
			}()
			wg.Wait()
		}
	}()

}
