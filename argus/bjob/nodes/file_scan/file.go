package file_scan

import (
	"context"
	"encoding/json"
	"fmt"
	"net/url"
	"strings"
	"time"

	"github.com/qiniu/xlog.v1"
	"qbox.us/qconf/qconfapi"
	"qiniupkg.com/api.v7/kodo"

	"qiniu.com/argus/argus/com/auth"
	"qiniu.com/argus/argus/com/bucket"
	ahttp "qiniu.com/argus/argus/com/http"
	"qiniu.com/argus/atserving/model"
	"qiniu.com/argus/bjob/com/records"
	job "qiniu.com/argus/bjob/proto"
	"qiniu.com/argus/bjob/workers"
)

type Request struct {
	UID   uint32 `json:"uid,omitempty"`
	Utype uint32 `json:"utype,omitempty"`

	Index struct {
		UID    uint32   `json:"uid,omitempty"`
		Bucket string   `json:"bucket"`
		Keys   []string `json:"keys"`
	}

	Params json.RawMessage `json:"params,omitempty"`

	Save *struct {
		UID    uint32 `json:"uid,omitempty"`
		Zone   int    `json:"zone,omitempty"`
		Bucket string `json:"bucket"`
		Prefix string `json:"prefix,omitempty"`
	} `json:"save,omitempty"`
}

var _ job.JobCreator = ScanNode{}

type ScanConfig struct {
	IsDomain bool            `json:"is_domain"`
	Qconf    qconfapi.Config `json:"qconf"`
	Kodo     kodo.Config     `json:"kodo"`
}
type ScanNode struct {
	ScanConfig
	URIFormat func(context.Context, uint32, string) string

	*qconfapi.Client
}

func NewScanNode(conf ScanConfig) ScanNode {
	node := ScanNode{ScanConfig: conf}
	node.Client = qconfapi.New(&node.Qconf)

	// 临时版本，依赖系统未做好 qiniu:/// 的支持
	node.URIFormat = func(ctx context.Context, uid uint32, uri string) string {
		xl := xlog.FromContextSafe(ctx)

		if !node.IsDomain {
			// Domain not available
			return uri
		}

		if !strings.HasPrefix(uri, "qiniu://") {
			return uri
		}
		_uri, err := url.Parse(uri)
		if err != nil {
			xl.Warnf("parse uri failed. %v %v", model.STRING(uri), err)
			return uri
		}
		strs := strings.SplitN(_uri.Path, "/", 3)
		if len(strs) < 3 {
			xl.Warnf("parse qiniu uri failed. %v", model.STRING(uri))
			return uri
		}
		bucket, key := strs[1], strs[2]

		ak, sk, err := auth.AkSk(node.Client, uid)
		if err != nil {
			xl.Warnf("get aksk failed. %d %v", uid, err)
			return uri
		}

		cli := ahttp.NewQiniuAuthRPCClient(ak, sk, time.Second*10)
		var domains = []struct {
			Domain string `json:"domain"`
			Tbl    string `json:"tbl"`
			Global bool   `json:"global"`
		}{}
		_ = cli.Call(context.Background(), &domains,
			"GET", fmt.Sprintf(node.Kodo.APIHost+"/v7/domain/list?tbl=%s", bucket),
		)
		if len(domains) == 0 {
			return uri
		}
		return kodo.New(0, &kodo.Config{AccessKey: ak, SecretKey: sk}).
			MakePrivateUrl(kodo.MakeBaseUrl(domains[0].Domain, key), nil)
	}
	return node
}

func (node ScanNode) NewMaster(ctx context.Context, reqBody []byte, env job.Env) (
	job.JobMaster, error) {

	var (
		req Request
		xl  = xlog.FromContextSafe(ctx)
	)
	if err := json.Unmarshal(reqBody, &req); err != nil {
		xl.Errorf("parse scan request error", err)
		return nil, err
	}

	xl.Infof("REQ: %+v", req)
	xl.Infof("Save: %+v", req.Save)

	var stg records.FileStorage
	{
		var (
			saveUID    uint32
			saveBucket string
			savePrefix string
		)

		if req.Save == nil || req.Save.Bucket == "" {
			saveBucket = req.Index.Bucket
			saveUID = req.Index.UID
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

		ak, sk, err := auth.AkSk(node.Client, saveUID)
		if err != nil {
			xl.Errorf("get aksk failed. %d %v", saveUID, err)
			return nil, err
		}

		stgPrefix := fmt.Sprintf("%s/%s/", savePrefix, env.JID)
		xl.Infof("StoragePrefix = %s", stgPrefix)

		stg = bucket.Bucket{
			Config: bucket.Config{Config: node.Kodo}.
				New(ak, sk, 0, saveBucket, stgPrefix),
		}
		// stg := records.NewLocalFS("") // FOR HACK
	}

	var scanner IScanner
	{
		var uid = req.Index.UID
		if uid == 0 {
			uid = req.UID
		}
		if uid == 0 {
			uid = env.UID
		}

		ak, sk, err := auth.AkSk(node.Client, uid)
		if err != nil {
			xl.Errorf("get aksk failed. %d %v", uid, err)
			return nil, err
		}

		scanner = Scanner{
			Config: Config{Config: node.Kodo}.New(
				ak, sk, req.Index.Bucket, req.Index.Keys),
		}
	}

	master := &ScanMaster{
		ScanConfig: node.ScanConfig,
		Request:    req,
		Env:        env,
		scanner:    scanner,
		URIFormat:  node.URIFormat,
		params:     req.Params,
		recordsStg: stg,
		records: records.NewRecords(ctx,
			records.NewFile(stg, time.Hour, 1000000),
			1024*1024*64),
	}

	return master, nil
}

type ScanMaster struct {
	ScanConfig
	Request
	job.Env

	scanner IScanner
	iter    IIter

	URIFormat func(context.Context, uint32, string) string
	params    json.RawMessage

	recordsStg records.FileStorage
	records    records.Records
}

func (m *ScanMaster) NextTask(ctx context.Context) ([]byte, string, bool) {
	xl := xlog.FromContextSafe(ctx)
	if m.iter == nil {
		m.iter, _ = m.scanner.Scan(ctx)
	}

	var (
		uid   uint32 = m.Request.UID
		utype uint32 = m.Request.Utype
	)
	if uid == 0 {
		uid = m.Env.UID
		utype = m.Env.Utype
	}

	itemURI := ""
	for {
		item, ok := m.iter.Next(ctx)
		xl.Infof("Get Item, %+v, %v", item, ok)
		if !ok {
			if err := m.iter.Error(); err != nil {
				xl.Warnf("Next Failed. %v", err)
			}
			return nil, "", false
		}

		itemURI = func() string {
			if m.URIFormat == nil {
				return item.URI
			}
			return m.URIFormat(ctx, uid, item.URI)
		}()

		has, err := m.records.HasKey(
			ctx, records.RecordKey(itemURI))
		if err != nil || !has {
			// 排重检查通过
			break
		}
	}

	bs, _ := json.Marshal(
		workers.InferenceImageTask{
			UID:    uid,
			Utype:  utype,
			URI:    itemURI,
			Params: m.params,
		})

	return bs, "", true
}
func (m ScanMaster) Error(ctx context.Context) error { return nil }
func (m *ScanMaster) Stop(ctx context.Context)       {}

func (m *ScanMaster) AppendResult(ctx context.Context, result job.TaskResult) error {
	var (
		xl    = xlog.FromContextSafe(ctx)
		_task workers.InferenceImageTask
	)
	_ = json.Unmarshal(result.Task().Value(ctx), &_task)

	xl.Infof("RET: %v %v", string(result.Value(ctx)), result.Error())

	if err := result.Error(); err != nil {
		_ = m.records.Append(ctx,
			records.RecordKey(_task.URI),
			records.RecordValue(err.Error()),
		)
	} else {
		_ = m.records.Append(ctx,
			records.RecordKey(_task.URI),
			records.RecordValue(result.Value(ctx)),
		)
	}
	return nil
}

func (m *ScanMaster) Result(ctx context.Context) ([]byte, error) {
	var xl = xlog.FromContextSafe(ctx)
	keys, err := m.records.Close(ctx)
	if err != nil {
		xl.Errorf("Close err, %+v", err)
		return nil, err
	}
	bytes, err := json.Marshal(
		struct {
			Keys []string `json:"keys"`
		}{Keys: keys},
	)
	xl.Infof("Result, %v, %+v", keys, err)
	return bytes, err
}
