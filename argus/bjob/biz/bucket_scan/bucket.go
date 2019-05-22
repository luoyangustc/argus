package bucket_scan

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"time"

	"github.com/qiniu/xlog.v1"
	"qbox.us/qconf/qconfapi"
	"qiniupkg.com/api.v7/kodo"
	log "qiniupkg.com/x/log.v7"

	"qiniu.com/argus/argus/com/auth"
	"qiniu.com/argus/argus/com/bucket"
	ahttp "qiniu.com/argus/argus/com/http"
	"qiniu.com/argus/bjob/com/records"
	job "qiniu.com/argus/bjob/proto"
)

type Request struct {
	UID    uint32 `json:"uid"`
	Utype  uint32 `json:"utype"`
	Zone   int    `json:"zone,omitempty"`
	Bucket string `json:"bucket"`
	Prefix string `json:"prefix,omitempty"`

	Save *struct {
		UID    uint32 `json:"uid,omitempty"`
		Zone   int    `json:"zone,omitempty"`
		Bucket string `json:"bucket"`
		Prefix string `json:"prefix,omitempty"`
	} `json:"save,omitempty"`
}

// var _ job.JobCreator = ScanNode{}

type ScanConfig struct {
	Qconf     qconfapi.Config `json:"qconf"`
	Kodo      kodo.Config     `json:"kodo"`
	BatchSize int             `json:"batch_size"`

	Hack *struct { // 本地Hack环境
		AK string `json:"ak"`
		SK string `json:"sk"`
	} `json:"hack,omitempty"`
}
type ScanNode struct {
	ScanConfig
	URIFormat func(uint32, int, string, string) string

	*qconfapi.Client
}

func NewScanNode(conf ScanConfig) ScanNode {
	node := ScanNode{ScanConfig: conf}
	if conf.Hack == nil {
		node.Client = qconfapi.New(&node.Qconf)
	}
	return node
}

func (node ScanNode) NewMaster(ctx context.Context, req Request, env job.Env) (
	*ScanMaster, error) {

	var (
		xl = xlog.FromContextSafe(ctx)
	)

	xl.Infof("REQ: %#v", req)
	xl.Infof("Save: %#v", req.Save)

	var stg records.FileStorage
	var scanStg records.FileStorage
	{
		var (
			saveUID    uint32
			saveBucket string
			savePrefix string
		)

		if req.Save == nil || req.Save.Bucket == "" {
			// xl.Warnf("no save. %v", req)
			// return nil, errors.New("no save")
			saveBucket = req.Bucket
			savePrefix = req.Prefix
		} else {
			saveUID = req.Save.UID
			saveBucket = req.Save.Bucket
			savePrefix = req.Save.Prefix
		}
		if saveUID == 0 {
			saveUID = req.UID
		}
		if saveUID == 0 {
			saveUID = env.UID
		}

		var ak, sk string
		if node.ScanConfig.Hack == nil {
			var err error
			ak, sk, err = auth.AkSk(node.Client, saveUID)
			if err != nil {
				xl.Errorf("get aksk failed. %d %v", saveUID, err)
				return nil, err
			}
		} else {
			ak, sk = node.ScanConfig.Hack.AK, node.ScanConfig.Hack.SK
		}

		stg = bucket.Bucket{
			Config: bucket.Config{Config: node.Kodo}.
				New(ak, sk, 0, saveBucket, savePrefix+"/"+env.JID+"/"),
		}
		scanStg = bucket.Bucket{
			Config: bucket.Config{Config: node.Kodo}.
				New(ak, sk, 0, saveBucket, savePrefix+"/scanneditem/"+env.JID+"/"),
		}
		// stg := records.NewLocalFS("") // FOR HACK
	}

	var ak, sk string
	if node.ScanConfig.Hack == nil {
		var err error
		ak, sk, err = auth.AkSk(node.Client, env.UID)
		if err != nil {
			xl.Errorf("get aksk failed. %d %v", env.UID, err)
			return nil, err
		}
	} else {
		ak, sk = node.ScanConfig.Hack.AK, node.ScanConfig.Hack.SK

		node.URIFormat = func(uid uint32, zone int, bucket, key string) string {

			domain, _ := func() (string, error) {
				cli := ahttp.NewQiniuAuthRPCClient(ak, sk, time.Second*10)
				var domains = []struct {
					Domain string `json:"domain"`
					Tbl    string `json:"tbl"`
					Global bool   `json:"global"`
				}{}
				_ = cli.Call(context.Background(), &domains,
					"GET", fmt.Sprintf("http://api.qiniu.com/v7/domain/list?tbl=%s", bucket),
				)

				log.Infof("%#v", domains)

				return domains[0].Domain, nil
			}()
			return kodo.New(0, &kodo.Config{AccessKey: ak, SecretKey: sk}).
				MakePrivateUrl(kodo.MakeBaseUrl(domain, key), &kodo.GetPolicy{Expires: 3600 * 24})
		}
	}

	master := &ScanMaster{
		ScanConfig: node.ScanConfig,
		Request:    req,
		Env:        env,
		bucket: bucket.Scanner{
			Config: bucket.Config{Config: node.Kodo}.
				New(ak, sk, req.Zone, req.Bucket, req.Prefix),
		},
		URIFormat: node.URIFormat,
		scanRecords: records.NewRecords(ctx,
			records.NewFile(scanStg, time.Hour, 1000000),
			0),
		recordsStg: stg,
		records: records.NewRecords(ctx,
			records.NewFile(stg, time.Hour, 1000000),
			1024*1024*60), // each element is 4bytes, so it'll cost about 300M memory
	}

	return master, nil
}

type ScanMaster struct {
	ScanConfig
	Request
	job.Env

	bucket bucket.IScanner
	iter   bucket.IIter

	URIFormat func(uint32, int, string, string) string

	recordsStg  records.FileStorage
	records     records.Records
	scanRecords records.Records
	order       int64
}

func (m *ScanMaster) NextTask(ctx context.Context) (string, string, bool) {
	if m.iter == nil {
		m.iter, _ = m.bucket.Scan(ctx, m.BatchSize)
	}

	var (
		xl         = xlog.FromContextSafe(ctx)
		uid uint32 = m.Request.UID
		// utype uint32 = m.Request.Utype
	)
	if uid == 0 {
		uid = m.Env.UID
		// utype = m.Env.Utype
	}

	var (
		count   int64 = 0
		itemURI       = ""
		kind    string
	)

	for {
		item, beginMarker, ok := m.iter.Next(ctx)
		if !ok {
			if err := m.iter.Error(); err != nil {
				xl.Warnf("Next Failed. %v", err)
			}
			return "", "", false
		}

		xl.Infof("ScannedItem: %+v, %s", item, beginMarker)

		// 获取URI
		itemURI = func() string {
			if m.URIFormat == nil {
				return fmt.Sprintf("qiniu://z%d/%s/%s", m.Zone, m.Bucket, item.Key)
			}
			return m.URIFormat(uid, m.Zone, m.Bucket, item.Key)
		}()

		kind = ""
		if strings.HasPrefix(item.MimeType, "image/") {
			kind = "image"
		} else if strings.HasPrefix(item.MimeType, "video/") {
			kind = "video"
			// } else if strings.HasPrefix(item.MimeType, "audio/") {
			// } else {
		}

		m.AppendScannedItem(ctx, itemURI)

		has, err := m.records.HasKey(
			ctx, records.RecordKey(itemURI))
		if err != nil || !has {
			// 排重检查通过
			break
		}

		count++
		if count%100000 == 0 {
			xl.Infof("skip. %d", count)
		}

	}

	return itemURI, kind, true
}
func (m ScanMaster) Error(ctx context.Context) error { return nil }
func (m *ScanMaster) Stop(ctx context.Context)       {}

func (m *ScanMaster) AppendScannedItem(ctx context.Context, uri string) {
	var xl = xlog.FromContextSafe(ctx)

	value, _ := json.Marshal(struct {
		TimeStamp int64 `json:"timestamp"`
		Order     int64 `json:"order"`
	}{
		Order:     m.order,
		TimeStamp: time.Now().UnixNano() / int64(time.Millisecond),
	})
	m.order++

	if err := m.scanRecords.Append(ctx,
		records.RecordKey(uri),
		records.RecordValue(value),
	); err != nil {
		xl.Errorf("Append ScannedItem err, %+v", err)
	}
}

func (m *ScanMaster) AppendResult(ctx context.Context, uri string, value []byte) error {
	// var xl = xlog.FromContextSafe(ctx)

	return m.records.Append(ctx,
		records.RecordKey(uri),
		records.RecordValue(value),
	)
}

func (m *ScanMaster) Result(ctx context.Context) ([]string, error) {
	var xl = xlog.FromContextSafe(ctx)
	_, err := m.scanRecords.Close(ctx)
	if err != nil {
		xl.Errorf("Close err, %+v", err)
		return nil, err
	}
	keys, err := m.records.Close(ctx)
	if err != nil {
		xl.Errorf("Close err, %+v", err)
		return nil, err
	}
	xl.Infof("Result, %v, %+v", keys, err)
	return keys, err
}
