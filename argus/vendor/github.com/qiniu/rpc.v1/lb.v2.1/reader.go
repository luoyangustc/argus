package lb

import (
	"io"
	"sync"
	"time"

	"github.com/qiniu/xlog.v1"
)

type SpeedLimit struct {
	CalcSpeedSizeThresholdB int64 `json:"calc_speed_size_threshold"` // 当某次请求返回 body 的长度大于 CalcSpeedSizeThresholdB 时启用低速熔断，避免小数据包干扰
	BanHostBelowBps         int64 `json:"ban_host_below_bps"`        // 当通过某个 host 访问的速度小于 BanHostBelowBps 时认为该 host 不可以访问，将该 host 加入屏蔽列表
}

type bodyReader struct {
	SpeedLimit
	Reader io.ReadCloser
	offset int64
	once   sync.Once
	tr     time.Duration
	h      *host
	xl     *xlog.Logger
}

func newBodyReader(xl *xlog.Logger, cfg SpeedLimit, rc io.ReadCloser, h *host) *bodyReader {
	r := new(bodyReader)
	r.SpeedLimit = cfg
	r.Reader = rc
	r.h = h
	if xl != nil {
		r.xl = xl
	} else {
		r.xl = xlog.NewDummy()
	}

	return r
}

func (r *bodyReader) Read(val []byte) (n int, err error) {
	start := time.Now()
	n, err = r.Reader.Read(val)
	r.tr += time.Now().Sub(start)
	r.offset += int64(n)
	if err != nil && r.offset > r.CalcSpeedSizeThresholdB {
		r.once.Do(func() {
			speed := float64(r.offset) / r.tr.Seconds()
			if speed < float64(r.BanHostBelowBps) {
				r.xl.Errorf("ban host(%s) below Bps: speed(%f) < banHostsBelowBps(%d)", r.h.raw, speed, r.BanHostBelowBps)
				r.h.SetFail(r.xl)
			}
		})
	}
	return
}

func (r *bodyReader) Close() error {
	return r.Reader.Close()
}
