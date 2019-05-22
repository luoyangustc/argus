package main

import (
	"context"
	"net"
	"net/http"
	"runtime"

	"github.com/prometheus/client_golang/prometheus/promhttp"

	cconf "qbox.us/cc/config"
	jsonlog "qbox.us/http/audit/jsonlog.v3"

	"github.com/qiniu/db/mgoutil.v3"
	"github.com/qiniu/errors"
	"github.com/qiniu/http/httputil.v1"
	"github.com/qiniu/http/restrpc.v1"
	"github.com/qiniu/http/servestk.v1"
	"github.com/qiniu/log.v1"
	"github.com/qiniu/xlog.v1"

	qdht "qbox.us/dht"
	"qiniu.com/argus/argus/com/dht"
	"qiniu.com/argus/atserving/config"
	"qiniu.com/argus/atserving/model"
	"qiniu.com/argus/com/pprof"
	FG "qiniu.com/argus/feature_group"
	IG "qiniu.com/argus/feature_group/imageg"
	STS "qiniu.com/argus/sts/client"
)

func main() {

	runtime.GOMAXPROCS(runtime.NumCPU())

	var (
		xl  = xlog.NewWith("main")
		ctx = xlog.NewContext(context.Background(), xl)
	)
	cconf.Init("f", "image_group", "image_group.conf")

	var conf = Config{}
	if err := cconf.Load(&conf.FileConfig); err != nil {
		xl.Fatal("Failed to load configure file!")
	}
	xl.Infof("load conf %#v", conf)
	if err := InitConfig(ctx, &conf); err != nil {
		xl.Fatalf("init config failed. %v", err)
	}
	log.SetOutputLevel(conf.DebugLevel)
	if _, err := conf.HTTPPort.Value(ctx); err != nil {
		xl.Fatalf("need http port. %v", err)
	}

	al, logf, err := jsonlog.Open("IG", &conf.FileConfig.AuditLog, nil)
	if err != nil {
		log.Fatal("jsonlog.Open failed:", errors.Detail(err))
	}
	defer logf.Close()

	mux := servestk.New(restrpc.NewServeMux(), al.Handler)
	mux.HandleFunc("GET /health", func(w http.ResponseWriter, r *http.Request) {
		w.Write([]byte("ok image-group"))
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

	var (
		featureAPIs FG.FeatureAPIs
		update      func(context.Context, qdht.NodeInfos)
		service     *IG.ImageGroupService
		sts         STS.Client
	)
	// 创建 STS.Client
	{
		sts = func() STS.Client {
			var vv, _ = conf.STSHosts.Value(ctx)
			var hosts = vv.([]model.ConfigSTSHost)
			var nodes = make(qdht.NodeInfos, 0, len(hosts))
			for _, host := range hosts {
				nodes = append(nodes, qdht.NodeInfo{Host: host.Host, Key: []byte(host.Key)})
			}
			lb, update := STS.NewLB(nodes, func() string { return xlog.GenReqId() }, nil)
			conf.STSHosts.Register(
				func(v interface{}) error {
					if v == nil {
						return nil
					}
					xl.Infof("sts update %#v", v)
					hosts := v.([]model.ConfigSTSHost)
					var nodes = make(qdht.NodeInfos, 0, len(hosts))
					for _, host := range hosts {
						nodes = append(nodes, qdht.NodeInfo{Host: host.Host, Key: []byte(host.Key)})
					}
					update(context.Background(), nodes)
					return nil
				},
			)
			return lb
		}()
	}
	{
		dfv := FG.FeatureVersion("v1") // default "v1"
		vv, _ := conf.FeatureVersion.Value(ctx)
		cfv := FG.FeatureVersion(vv.(string))
		xl.Infof("feature version. %s %s", dfv, cfv)
		f := make(map[FG.FeatureVersion]FG.FeatureAPI)
		ks, vs, _ := conf.FeatureAPIs.Values(ctx)
		for i, k := range ks {
			xl.Infof("feature api. %v %#v", k, vs[i])
			config := vs[i].(IG.ImageGFeatureAPIConfig)
			f[FG.FeatureVersion(k.(string))] = IG.NewImageGFeatureAPI(config)
		}
		featureAPIs = FG.NewFeatureAPIs(f, dfv, cfv)

		conf.FeatureVersion.Register(
			func(v interface{}) error {
				if v == nil {
					return nil
				}
				xl.Infof("UpdateConfig current feature version. %v", v)
				err := featureAPIs.SetCurrent(FG.FeatureVersion(v.(string)))
				return err
			},
		)

		conf.FeatureAPIs.Register(
			func(k, v interface{}) error {
				name := FG.FeatureVersion(k.(string))
				xl.Infof("UpdateConfig feature api. %v %#v", name, v)
				if v == nil {
					featureAPIs.Reset(name, nil)
				} else {
					config := v.(IG.ImageGFeatureAPIConfig)
					featureAPIs.Reset(name, IG.NewImageGFeatureAPI(config))
				}
				return nil
			},
		)
	}
	{
		vv, _ := conf.Mgo.Value(ctx)
		mgo := vv.(mgoutil.Config)
		xl.Info("mongo conf", config.DumpJsonConfig(mgo))
		hub, err := FG.NewHubInMgo(&mgo,
			&struct {
				Hubs     mgoutil.Collection `coll:"ig_hub_hubs"`
				Features mgoutil.Collection `coll:"ig_hub_features"`
			}{},
		)
		if err != nil {
			xl.Fatalf("NewHubInMgo fails. %v", err)
		}
		m, err := IG.NewImageGroupManagerInMgo(&mgo, hub)
		if err != nil {
			xl.Fatalf("NewImageGroupManagerInMgo fails. %v", err)
		}

		vv2, _ := conf.Search.Value(ctx)
		_config := vv2.(IG.Config)
		xl.Info("ig conf", config.DumpJsonConfig(_config))
		service, update = IG.NewImageGroupServic(_config, sts, m, featureAPIs)

		d := dht.NewETCD(client, IG.IMAGE_GROUP_WORKER_NODE_PREFIX)
		nodes, err := d.Nodes(ctx, func(unodes qdht.NodeInfos) error {
			xl.Infof("etcd update nodes. %v", unodes)
			update(ctx, unodes)
			return nil
		})
		if err != nil {
			xl.Fatalf("etcd nodes get fail. %v", err)
		} else {
			xl.Infof("etcd nodes. %v", nodes)
			update(ctx, nodes)
		}
	}

	{
		router := restrpc.Router{
			PatternPrefix: "v1",
			Mux:           mux,
		}
		router.Register(service)
	}

	if err = http.ListenAndServe(
		net.JoinHostPort("0.0.0.0",
			config.GetString(ctx, conf.HTTPPort, "80")),
		mux); err != nil {
		log.Fatal("http.ListenAndServe", err)
	}
}
