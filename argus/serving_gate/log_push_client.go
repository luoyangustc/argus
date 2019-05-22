package gate

import (
	"bytes"
	"context"
	"encoding/base64"
	"io"
	"io/ioutil"
	"net/http"
	"strings"
	"sync"
	"time"

	"github.com/pkg/errors"
	xlog "github.com/qiniu/xlog.v1"
	"github.com/ugorji/go/codec"
	STS "qiniu.com/argus/sts/client"
)

const (
	DataURIPrefix      = "data:application/octet-stream;base64,"
	maxLogPushFileSize = 10 * 1024 * 1024
	maxLogPushMsgSize  = 20 * 1024 * 1024
)

type LogPushClient struct {
	cfg    LogPushConfig
	client *http.Client
	sts    STS.Client
	mux    sync.RWMutex
}

type LogPushConfig struct {
	URL           string  `json:"url"`
	Open          bool    `json:"open"`
	TimeoutSecond float64 `json:"timeout_second"`
}

func NewLogPushClient(cfg LogPushConfig, client *http.Client, sts STS.Client) *LogPushClient {
	return &LogPushClient{cfg: cfg, client: client, sts: sts}
}

func (l *LogPushClient) UpdateConfig(cfg LogPushConfig) {
	l.mux.Lock()
	l.cfg = cfg
	l.mux.Unlock()
}

func (l *LogPushClient) getConfig() (cfg LogPushConfig) {
	l.mux.RLock()
	cfg = l.cfg
	l.mux.RUnlock()
	return
}

// 发送日志
func (l *LogPushClient) sendRawLog(al *aiprdLog) (err error) {
	cfg := l.getConfig()
	if !cfg.Open {
		return
	}
	start := time.Now()
	status := ""
	defer func() {
		if status == "" && err != nil {
			status = err.Error()
		}
		logPushResponseTime.WithLabelValues(status).Observe(float64(time.Since(start)) / 1e9)
	}()
	b := &bytes.Buffer{}
	codec.NewEncoder(b, &codec.MsgpackHandle{}).MustEncode(al)
	al.URIData = nil
	logPushSendBytes.Add(float64(b.Len()))
	req, err := http.NewRequest("POST", cfg.URL, b)
	if err != nil {
		return errors.Wrap(err, "logPushClient http.NewRequest")
	}
	req.Header.Add("Content-Type", "application/x-msgpack")
	timeout := cfg.TimeoutSecond
	if timeout == 0 {
		timeout = 3
	}
	ctx, cancel := context.WithTimeout(context.Background(), time.Second*time.Duration(timeout))
	defer cancel()
	resp, err := l.client.Do(req.WithContext(ctx))
	if err != nil {
		return errors.Wrap(err, "logPushClient http.Do")
	}
	defer resp.Body.Close()
	io.Copy(ioutil.Discard, resp.Body)
	if resp.StatusCode != http.StatusOK {
		status = resp.Status
		return errors.Errorf("logPush bad status:%s", resp.Status)
	}
	return nil
}

// 自动去sts拉文件然后发送日志
func (l *LogPushClient) sendLog(ctx context.Context, al *aiprdLog) {
	logPushProcessing.Inc()
	defer logPushProcessing.Dec()
	xl := xlog.FromContextSafe(ctx)
	if !l.getConfig().Open {
		return
	}
	al.URIData = make([][]byte, len(al.uri))
	sizeCnt := 0
	for i, uri := range al.uri {
		func() {
			if strings.HasPrefix(uri, DataURIPrefix) {
				buf, _ := base64.StdEncoding.DecodeString(uri[len(DataURIPrefix):])
				al.URIData[i] = buf
				return
			}
			s, _, _, err := l.sts.Get(ctx, uri, nil)
			if err != nil {
				xl.Error("fetch file from sts", uri, err)
				return
			}
			defer s.Close()
			buf, err := ioutil.ReadAll(io.LimitReader(s, maxLogPushFileSize+1))
			if err != nil {
				xl.Error("read file from sts", uri, err)
				return
			}
			resourceSize.WithLabelValues(al.AiprdLog.Cmd).Observe(float64(len(buf)))
			if len(buf) > maxLogPushFileSize {
				xl.Warn("LogPushClient: file to large, skip it", uri, err)
				logPushSkipFile.Inc()
				return
			}

			sizeCnt += len(buf)
			if sizeCnt > maxLogPushMsgSize {
				xl.Warn("LogPushClient: file size cnt to large, skip it", uri, err)
				logPushSkipFile.Inc()
				return
			}
			al.URIData[i] = buf
		}()
	}

	logPushMemory.Add(float64(sizeCnt))
	defer logPushMemory.Sub(float64(sizeCnt))

	if err := l.sendRawLog(al); err != nil {
		xl.Error("send log error", err)
	}
}
