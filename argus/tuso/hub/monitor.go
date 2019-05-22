package hub

import (
	"context"
	"fmt"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/qiniu/db/mgoutil.v3"
)

const (
	namespace             = "ava"
	subsystem             = "tuso_so"
	monitorAggregateLimit = 10000
)

var opLogCnt = prometheus.NewGaugeVec(prometheus.GaugeOpts{
	Namespace: namespace,
	Subsystem: subsystem,
	Name:      "oplog_cnt",
	Help:      "OpLog 条目分类统计",
}, []string{"field"})

var collectionCnt = prometheus.NewGaugeVec(prometheus.GaugeOpts{
	Namespace: namespace,
	Subsystem: subsystem,
	Name:      "collection_cnt",
	Help:      "mongodb 文档数目统计",
}, []string{"collection"})

func init() {
	prometheus.MustRegister(opLogCnt, collectionCnt)
}

func (s *opLogProcess) monitor(ctx context.Context) {
	for _, field := range []string{"op", "status", "hub_name"} {
		r, err := s.db.opLogStat(monitorAggregateLimit, field)
		if err != nil {
			xl.Warn("db.opLogStat", monitorAggregateLimit, field, err)
		} else {
			for _, v := range r {
				opLogCnt.WithLabelValues(fmt.Sprintf("%s_%s", field, v.FieldValue)).Set(float64(v.Cnt))
			}
		}
	}
	for _, col := range []mgoutil.Collection{s.db.Hub, s.db.OpLog, s.db.HubMeta, s.db.FileMeta} {
		n, err := col.Count()
		if err != nil {
			xl.Warn("db.count", col.FullName, err)
		} else {
			collectionCnt.WithLabelValues(col.FullName).Set(float64(n))
		}
	}
}

func (s *opLogProcess) monitorLoop(ctx context.Context) {
	for {
		select {
		case <-ctx.Done():
			return
		default:
		}
		s.monitor(ctx)
		time.Sleep(time.Second * 20)
	}
}
