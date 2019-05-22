package serving

import (
	"context"
	"encoding/base64"
	"encoding/binary"
	"errors"
	"fmt"
	"net/http"
	"time"

	"github.com/qiniu/rpc.v3"
)

type FSCreateReq struct {
	Name      string `json:"name,omitempty"`
	Dimension int    `json:"dimension"`
	Precision int    `json:"precision"`
	Size      int    `json:"size"`
	Version   uint64 `json:"version"`
	State     int    `json:"state"`
	Timeout   int64  `json:"timeout,omitempty"`
}

type FSGetResp struct {
	Dimension int    `json:"dimension"`
	Precision int    `json:"precision"`
	Size      int    `json:"size"`
	Version   uint64 `json:"version"`
	State     int    `json:"state"`
}

type Feature struct {
	ID    string `json:"id,omitempty" bson:"id"`
	Name  string `json:"name,omitempty" bson:"name,omitempty"`
	Value []byte `json:"Value" bson:"Value"`
}

type FSAddReq struct {
	Name     string    `json:"name,omitempty"`
	Features []Feature `json:"features"`
}

type FSDeleteReq struct {
	Name string   `json:"name,omitempty"`
	IDs  []string `json:"ids"`
}

type FSDeleteResp struct {
	Deleted []string `json:"deleted"`
}

type FSSearchReq struct {
	Name      string    `json:"name,omitempty"`
	Features  []Feature `json:"features"`
	Threshold float32   `json:"threshold"`
	Limit     int       `json:"limit"`
}

type float32Feature struct {
	ID    string `json:"id,omitempty"`
	Name  string `json:"name,omitempty"`
	Value string `json:"value"`
}
type FSSearchResp struct {
	SearchResults [][]struct {
		Score float32 `json:"score"`
		ID    string  `json:"id,omitempty"`
	} `json:"search_results"`
}

//----------------------------------------------------------------------------//
type FeatureSearch interface {
	Create(context.Context, FSCreateReq) error
	Get(context.Context, string) (FSGetResp, error)
	Destroy(context.Context, string) error
	Add(context.Context, FSAddReq) error
	Delete(context.Context, FSDeleteReq) (FSDeleteResp, error)
	Search(context.Context, FSSearchReq) (FSSearchResp, error)
	UpdateState(context.Context, string, int) error
}

type _FeatureSearch struct {
	url     string
	timeout time.Duration
	*rpc.Client
}

func NewFeatureSearch(conf EvalConfig) _FeatureSearch {
	url := conf.Host + "/v1"
	if conf.URL != "" {
		url = conf.URL
	}
	return _FeatureSearch{url: url, timeout: time.Duration(conf.Timeout) * time.Second}
}

func bigEndianToLittleEndian(a []byte) []byte {
	b := make([]byte, len(a))
	for i := 0; i < len(a); i += 4 {
		r := binary.BigEndian.Uint32(a[i : i+4])
		binary.LittleEndian.PutUint32(b[i:], r)
	}
	return b
}

func (fs _FeatureSearch) eval(ctx context.Context, method, uri string, req interface{}, resp interface{}) (err error) {
	var (
		client *rpc.Client
	)
	if fs.Client == nil {
		client = NewDefaultStubRPCClient(fs.timeout)
	} else {
		client = fs.Client
	}
	err = callRetry(ctx,
		func(ctx context.Context) error {
			var err1 error
			err1 = client.CallWithJson(ctx, resp, method, fs.url+uri, req)
			return err1
		})
	return
}

func (fs _FeatureSearch) Create(
	ctx context.Context, req FSCreateReq,
) (err error) {
	uri := fmt.Sprintf("/sets/%s", req.Name)
	req.Name = ""
	return fs.eval(ctx, "POST", uri, &req, nil)
}

func (fs _FeatureSearch) Get(
	ctx context.Context, name string,
) (resp FSGetResp, err error) {
	uri := fmt.Sprintf("/sets/%s", name)
	err = fs.eval(ctx, "GET", uri, nil, &resp)
	if rpc.HttpCodeOf(err) == http.StatusNotFound {
		err = nil
	}
	return
}

func (fs _FeatureSearch) UpdateState(
	ctx context.Context, name string, state int,
) (err error) {
	uri := fmt.Sprintf("/sets/%s/state/%d", name, state)
	err = fs.eval(ctx, "POST", uri, nil, nil)
	return
}

func (fs _FeatureSearch) Add(
	ctx context.Context, req FSAddReq,
) (err error) {
	uri := fmt.Sprintf("/sets/%s/add", req.Name)
	req.Name = ""
	var request struct {
		Feautres []float32Feature `json:"features"`
	}
	for _, feature := range req.Features {
		if len(feature.Value) != 512*4 {
			err = errors.New("invalid feature")
			return
		}
		ff := float32Feature{Name: feature.Name, ID: feature.ID, Value: base64.StdEncoding.EncodeToString(bigEndianToLittleEndian(feature.Value))}
		request.Feautres = append(request.Feautres, ff)
	}

	err = fs.eval(ctx, "POST", uri, request, nil)
	return
}

func (fs _FeatureSearch) Delete(
	ctx context.Context, req FSDeleteReq,
) (resp FSDeleteResp, err error) {
	uri := fmt.Sprintf("/sets/%s/delete", req.Name)
	req.Name = ""
	if len(req.IDs) > 0 {
		err = fs.eval(ctx, "POST", uri, req, &resp)
	}
	return
}

func (fs _FeatureSearch) Search(
	ctx context.Context, req FSSearchReq,
) (resp FSSearchResp, err error) {

	uri := fmt.Sprintf("/sets/%s/search", req.Name)
	req.Name = ""
	if req.Limit == 0 {
		req.Limit = 1
	}

	var request struct {
		Feautres  [][]byte `json:"features"`
		Threshold float32  `json:"threshold"`
		Limit     int      `json:"limit"`
	}
	for _, feature := range req.Features {
		if len(feature.Value) != 512*4 {
			err = errors.New("invalid feature")
			return
		}
		request.Feautres = append(request.Feautres, bigEndianToLittleEndian(feature.Value))

	}
	request.Limit = req.Limit
	request.Threshold = req.Threshold
	if len(request.Feautres) > 0 {
		err = fs.eval(ctx, "POST", uri, request, &resp)
	}
	return
}

func (fs _FeatureSearch) Destroy(
	ctx context.Context, name string,
) error {

	uri := fmt.Sprintf("/sets/%s/destroy", name)
	return fs.eval(ctx, "POST", uri, nil, nil)
}
