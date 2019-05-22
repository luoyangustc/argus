package main

import (
	"context"
	"net/http"

	"github.com/gogo/protobuf/proto"

	zmq "github.com/pebbe/zmq4"
	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promhttp"
	pb "qiniu.com/ai-sdk/proto"
)

const NAMESPACE string = "ava"
const SUBSYSTEM string = "serving_eval"

func monitorRegister() (*prometheus.HistogramVec, *prometheus.HistogramVec, *prometheus.HistogramVec) {

	// eval时长，不包括下载，包括预处理和推理
	var evalTime = prometheus.NewHistogramVec(
		prometheus.HistogramOpts{
			Namespace: NAMESPACE,
			Subsystem: SUBSYSTEM,
			Name:      "eval_time",
			Help:      "eval time of requests",
			Buckets: []float64{
				0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
				1, 2, 3, 4, 5, 6, 7, 8, 9,
				10, 20, 30, 40, 50, 60,
			}, // time.second
			ConstLabels: map[string]string{"app": cfg.App},
		},
		[]string{"pid", "http_code"},
	)

	// inference 推理时长，eval端观察到的时间
	var forwardTime = prometheus.NewHistogramVec(
		prometheus.HistogramOpts{
			Namespace: NAMESPACE,
			Subsystem: SUBSYSTEM,
			Name:      "forward_time",
			Help:      "forward time of requests",
			Buckets: []float64{
				0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
				1, 2, 3, 4, 5, 6, 7, 8, 9,
				10, 20, 30, 40, 50, 60,
			}, // time.second
			ConstLabels: map[string]string{"app": cfg.App},
		},
		[]string{"pid"},
	)

	var batchSize = prometheus.NewHistogramVec(
		prometheus.HistogramOpts{
			Namespace: NAMESPACE,
			Subsystem: SUBSYSTEM,
			Name:      "batch_size",
			Help:      "batch_size of requests",
			Buckets: []float64{
				1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 32, 64, 128,
			},
			ConstLabels: map[string]string{"app": cfg.App},
		},
		[]string{"pid"},
	)

	prometheus.MustRegister(evalTime, forwardTime, batchSize)
	return evalTime, forwardTime, batchSize
}

func monitorCollecter(ctx context.Context) {
	evalTime, forwardTime, batchSize := monitorRegister()
	monitor, err := zmq.NewSocket(zmq.PULL)
	ce(err)
	ce(monitor.Bind(MONIROT_ZMQ_ADDR))
	var msg pb.MonitorMetric
	for {
		select {
		case <-ctx.Done():
			return
		default:
		}
		buf, err := monitor.RecvBytes(0)
		if err != nil {
			xl.Warn("reqer recv", err)
			continue
		}
		err = proto.Unmarshal(buf, &msg)
		ce(err)
		xl.Debugf("monitor %#v", msg)
		switch msg.Kind {
		case "inference_started_success":
			globalSubprocessStatus.inferenceStartSuccess()
			// fallthrough
		case "forward_started_success":
			globalSubprocessStatus.forwardStartSuccess()
			// xl.Info("subprocess started ok", msg.Kind, msg.Pid)
		case "eval_time":
			evalTime.WithLabelValues(msg.Pid, msg.Code).Observe(msg.Value)
		case "forward_time":
			forwardTime.WithLabelValues(msg.Pid).Observe(msg.Value)
		case "batch_size":
			batchSize.WithLabelValues(msg.Pid).Observe(msg.Value)
		default:
			panic("bad monitor msg")
		}
	}
}

func runMonitor(ctx context.Context, addr string) {
	srv := http.NewServeMux()
	srv.Handle("/metrics", promhttp.Handler())
	go monitorCollecter(ctx)
	globalSubprocessStatus.waitStart()
	ce(http.ListenAndServe(addr, srv))
}
