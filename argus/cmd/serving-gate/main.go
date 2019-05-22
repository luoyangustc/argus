package main

import (
	"context"
	"net"
	"net/http"
	"runtime"

	"github.com/prometheus/client_golang/prometheus/promhttp"

	cconf "qbox.us/cc/config"
	"qbox.us/dht"
	jsonlog "qbox.us/http/audit/jsonlog.v3"

	"github.com/qiniu/errors"
	"github.com/qiniu/http/httputil.v1"
	"github.com/qiniu/http/restrpc.v1"
	"github.com/qiniu/http/servestk.v1"
	"github.com/qiniu/log.v1"
	"github.com/qiniu/xlog.v1"

	"qiniu.com/argus/atserving/config"
	"qiniu.com/argus/atserving/model"
	"qiniu.com/argus/atserving/simplequeue/sq"
	"qiniu.com/argus/com/pprof"
	"qiniu.com/argus/serving_gate"
	STS "qiniu.com/argus/sts/client"
)

func main() {

	runtime.GOMAXPROCS(runtime.NumCPU())

	var (
		xl  = xlog.NewWith("main")
		ctx = xlog.NewContext(context.Background(), xl)
	)
	cconf.Init("f", "serving-gate", "serving-gate.conf")

	if RUN_MODE == _ModeStandalone {
		main4Standalone(ctx)
		return
	}

	var (
		conf = Config{}

		sts      STS.Client
		producer gate.Producer
		worker   gate.Worker
		evals    gate.Evals
		logPush  *gate.LogPushClient
	)
	if err := cconf.Load(&conf.FileConfig); err != nil {
		xl.Fatal("Failed to load configure file!")
	}
	if err := InitConfig(ctx, &conf); err != nil {
		xl.Fatalf("init config failed. %v", err)
	}
	xl.Infof("load conf %#v", conf)
	if _, err := conf.HTTPPort.Value(ctx); err != nil {
		xl.Fatalf("need http port. %v", err)
	}

	if host, err := conf.CallBackHost.Value(ctx); err != nil {
		xl.Fatalf("need callback host. %v", err)
	} else {
		worker = gate.NewWorker("http://" + host.(string))
	}

	{
		var hosts []model.ConfigSTSHost
		if vv, err := conf.STSHosts.Value(ctx); err != nil {
			xl.Fatalf("load sts hosts failed. %v", err)
		} else {
			hosts = vv.([]model.ConfigSTSHost)
		}
		sts = func() STS.Client {
			var nodes = make(dht.NodeInfos, 0, len(hosts))
			for _, host := range hosts {
				nodes = append(nodes, dht.NodeInfo{Host: host.Host, Key: []byte(host.Key)})
			}
			lb, update := STS.NewLB(nodes, func() string { return xlog.GenReqId() }, nil)
			conf.STSHosts.Register(
				func(v interface{}) error {
					if v == nil {
						return nil
					}
					xl.Infof("sts update %#v", v)
					hosts := v.([]model.ConfigSTSHost)
					var nodes = make(dht.NodeInfos, 0, len(hosts))
					for _, host := range hosts {
						nodes = append(nodes, dht.NodeInfo{Host: host.Host, Key: []byte(host.Key)})
					}
					update(context.Background(), nodes)
					return nil
				},
			)
			return lb
		}()
	}
	{
		var sqHosts []string
		if vv, err := conf.SQHosts.Value(ctx); err != nil {
			xl.Fatalf("load sq hosts failed. %v", err)
		} else {
			sqHosts = vv.([]string)
		}

		producer = func() gate.Producer {
			var ps = make([]*sq.Producer, 0, len(sqHosts))
			for _, host := range sqHosts {
				p, _ := sq.NewProducer(host) // TODO err
				ps = append(ps, p)
			}
			return gate.NewProducer(ps)
		}()
	}

	{
		evals = gate.NewEvals()
		conf.Workers.Register(
			func(k1, v1 interface{}) error {
				var k2 = k1.(model.ConfigKeyWorker)
				if v1 == nil {
					if k2.App == nil {
					} else {
						evals.UnsetWorker(*k2.App, k2.Version)
					}
				} else {
					var v2 = v1.(model.ConfigWorker)
					if k2.App == nil {
						evals.SetWorkerDefault(v2)
					} else {
						evals.SetWorker(*k2.App, k2.Version, v2)
					}
				}
				return nil
			},
		)
		{
			ks, vs, _ := conf.Workers.Values(ctx)
			for i, k1 := range ks {
				var k2 = k1.(model.ConfigKeyWorker)
				var v2 = vs[i].(model.ConfigWorker)
				if k2.App == nil {
					evals.SetWorkerDefault(v2)
				} else {
					evals.SetWorker(*k2.App, k2.Version, v2)
				}
			}
		}
		conf.AppMetadataDefault.Register(
			func(v1 interface{}) error {
				var v2 = v1.(model.ConfigAppMetadata)
				evals.SetAppMetadataDefault(v2)
				return nil
			},
		)
		{
			v1, err := conf.AppMetadataDefault.Value(ctx)
			if err != nil {
				xl.Infof("no app default metadata config. %v", err)
			} else {
				xl.Infof("App Default Metadata: %#v", v1)
				evals.SetAppMetadataDefault(v1.(model.ConfigAppMetadata))
			}
		}
		conf.AppMetadatas.Register(
			func(k1, v1 interface{}) error {
				var k2 = k1.(model.ConfigKeyAppMetadata)
				if v1 == nil {
					evals.UnsetAppMetadata(k2.App)
				} else {
					var v2 = v1.(model.ConfigAppMetadata)
					evals.SetAppMetadata(k2.App, v2)
				}
				return nil
			},
		)
		{
			ks, vs, _ := conf.AppMetadatas.Values(ctx)
			for i, k1 := range ks {
				var k2 = k1.(model.ConfigKeyAppMetadata)
				evals.SetAppMetadata(k2.App, vs[i].(model.ConfigAppMetadata))
			}
		}
		conf.AppReleases.Register(
			func(k1, v1 interface{}) error {
				var k2 = k1.(model.ConfigKeyAppRelease)
				if v1 == nil {
					evals.Unregister(k2.App, k2.Version)
				} else {
					var v2 = v1.(model.ConfigAppRelease)
					evals.Register(k2.App, k2.Version, v2)
				}
				return nil
			},
		)
		{
			ks, vs, _ := conf.AppReleases.Values(ctx)
			for i, k1 := range ks {
				var k2 = k1.(model.ConfigKeyAppRelease)
				evals.Register(k2.App, k2.Version, vs[i].(model.ConfigAppRelease))
			}
		}
	}

	{
		vv, err := conf.LogPush.Value(ctx)
		if err != nil {
			xl.Warnf("load logpush hosts failed. %v", err)
			logPush = gate.NewLogPushClient(gate.LogPushConfig{Open: false}, http.DefaultClient, sts)
		} else {
			logPush = gate.NewLogPushClient(vv.(gate.LogPushConfig), http.DefaultClient, sts)
			conf.LogPush.Register(
				func(v interface{}) error {
					if v == nil {
						return nil
					}
					xl.Infof("LogPush UpdateConfig %#v", v)
					logPush.UpdateConfig(v.(gate.LogPushConfig))
					return nil
				},
			)
		}
	}

	var _gate = gate.NewGate(producer, worker, sts, evals)

	server(
		config.GetString(ctx, conf.HTTPPort, "80"),
		&conf.AuditLog,
		conf,
		logPush, evals, _gate,
		func(mux *servestk.ServeStack) {
			router := &restrpc.Router{
				PatternPrefix: "",
				Factory:       restrpc.Factory,
				Mux:           mux,
			}
			router.Register(worker)
		},
	)

}

func server(
	port string,
	auditConfig *jsonlog.Config,
	conf interface{},
	logPush *gate.LogPushClient, // TODO
	evals gate.Evals, _gate gate.Gate,
	more func(*servestk.ServeStack),
) {

	var server = gate.NewServer(_gate, evals, logPush)

	al, logf, err := jsonlog.Open("SGATE", auditConfig, nil)
	if err != nil {
		log.Fatal("jsonlog.Open failed:", errors.Detail(err))
	}
	defer logf.Close()

	mux := servestk.New(restrpc.NewServeMux(), al.Handler)
	mux.HandleFunc("GET /health", func(w http.ResponseWriter, r *http.Request) {
		w.Write([]byte("ok serving-gate"))
	})
	mux.Handle("GET /metrics", promhttp.Handler())
	mux.HandleFunc("GET /configs", func(w http.ResponseWriter, r *http.Request) {
		m, err := config.DumpSimpleStruct(context.Background(), conf)
		if err != nil {
			httputil.ReplyErr(w, http.StatusInternalServerError, err.Error())
		} else {
			httputil.Reply(w, http.StatusOK, m)
		}
	})

	{
		mux2 := http.NewServeMux()
		mux.SetDefault(mux2)
		pprof.Handle(mux2, "")
	}

	router := &restrpc.Router{
		PatternPrefix: "v1",
		Factory:       restrpc.Factory,
		Mux:           mux,
	}
	router.Register(server)

	if more != nil {
		more(mux)
	}

	if err = http.ListenAndServe(net.JoinHostPort("0.0.0.0", port), mux); err != nil {
		log.Fatal("http.ListenAndServe", err)
	}
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
