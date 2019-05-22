package bucket_censor

import (
	"context"
	"encoding/json"

	httputil "github.com/qiniu/http/httputil.v1"
	"github.com/qiniu/xlog.v1"

	BS "qiniu.com/argus/bjob/biz/bucket_scan"
	job "qiniu.com/argus/bjob/proto"
	workers "qiniu.com/argus/bjob/workers/censor"
)

type Request struct {
	UID    uint32 `json:"uid"`
	Utype  uint32 `json:"utype"`
	Zone   int    `json:"zone,omitempty"`
	Bucket string `json:"bucket"`
	Prefix string `json:"prefix,omitempty"`

	// Params    interface{} `json:"params,omitempty"`
	MimeTypes []string `json:"mimetypes,omitempty"`
	Params    struct {
		Image json.RawMessage `json:"image,omitempty"`
		Video json.RawMessage `json:"video,omitempty"`
	} `json:"params"`

	Save *struct {
		UID    uint32 `json:"uid,omitempty"`
		Zone   int    `json:"zone,omitempty"`
		Bucket string `json:"bucket"`
		Prefix string `json:"prefix,omitempty"`
	} `json:"save,omitempty"`
}

type ScanConfig BS.ScanConfig

var _ job.JobCreator = ScanNode{}

type ScanNode struct {
	BS.ScanNode
}

func NewScanNode(conf ScanConfig) ScanNode {
	return ScanNode{ScanNode: BS.NewScanNode(BS.ScanConfig(conf))}
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

	xl.Infof("REQ: %#v", req)
	xl.Infof("Save: %#v", req.Save)

	master, err := node.ScanNode.NewMaster(ctx,
		BS.Request{
			UID: req.UID, Utype: req.Utype,
			Zone: req.Zone, Bucket: req.Bucket, Prefix: req.Prefix,
			Save: req.Save,
		},
		env)
	if err != nil {
		return nil, err
	}

	master0 := &ScanMaster{
		ScanMaster: master,
		Request:    req,
	}

	if len(req.MimeTypes) == 0 {
		master0.MimeTypes = []string{"image", "video"}
	} else {
		master0.MimeTypes = req.MimeTypes
	}
	master0.ImageParams = req.Params.Image
	master0.VideoParams = req.Params.Video

	return master0, nil
}

type ScanMaster struct {
	*BS.ScanMaster
	Request

	MimeTypes   []string
	ImageParams json.RawMessage
	VideoParams json.RawMessage
}

func (m *ScanMaster) NextTask(ctx context.Context) ([]byte, string, bool) {
	for {
		uri, kind, ok := m.ScanMaster.NextTask(ctx)
		if !ok {
			return nil, "", ok
		}

		var found = false
		for _, kid := range m.MimeTypes {
			if kid == kind {
				found = true
				break
			}
		}
		if !found {
			continue
		}

		var (
			uid   uint32 = m.Request.UID
			utype uint32 = m.Request.Utype
		)
		if uid == 0 {
			uid = m.Env.UID
			utype = m.Env.Utype
		}

		switch kind {
		case "image":
			bs, _ := json.Marshal(
				workers.Task{
					UID:      uid,
					Utype:    utype,
					URI:      uri,
					Mimetype: kind,
					Params:   m.ImageParams,
				})

			return bs, kind, true
		case "video":
			bs, _ := json.Marshal(
				workers.Task{
					UID:      uid,
					Utype:    utype,
					URI:      uri,
					Mimetype: kind,
					Params:   m.VideoParams,
				})
			return bs, kind, true
		}
	}
}
func (m ScanMaster) Error(ctx context.Context) error { return nil }
func (m *ScanMaster) Stop(ctx context.Context)       {}

func (m *ScanMaster) AppendResult(ctx context.Context, result job.TaskResult) error {
	var (
		xl    = xlog.FromContextSafe(ctx)
		_task workers.Task
	)
	_ = json.Unmarshal(result.Task().Value(ctx), &_task)
	var ret = struct {
		Code     int             `json:"code"`
		Mimetype string          `json:"mimetype"`
		Error    string          `json:"error,omitempty"`
		Result   json.RawMessage `json:"result,omitempty"`
	}{Mimetype: _task.Mimetype}

	xl.Infof("RET: %v %v", string(result.Value(ctx)), result.Error())
	if err := result.Error(); err == nil {
		ret.Code = 200
		ret.Result = result.Value(ctx)
	} else {
		ret.Code, ret.Error = httputil.DetectError(err)
	}
	bs, _ := json.Marshal(ret)
	_ = m.ScanMaster.AppendResult(ctx, _task.URI, bs)
	return nil
}

func (m *ScanMaster) Result(ctx context.Context) ([]byte, error) {
	keys, err := m.ScanMaster.Result(ctx)
	if err != nil {
		return nil, err
	}
	bytes, err := json.Marshal(
		struct {
			Keys []string `json:"keys"`
		}{Keys: keys},
	)
	return bytes, err
}
