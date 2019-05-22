package idcensor

import (
	"context"
	"encoding/json"
	"net/http"
	"time"

	"qbox.us/net/httputil"

	"github.com/qiniu/rpc.v1"
	"github.com/qiniu/xlog.v1"
	"qiniu.com/argus/utility"
)

type iFaceSim interface {
	Eval(context.Context, *utility.FaceSimReq) (utility.FaceSimResp, error)
}

type _FaceSim struct {
	host string
	*rpc.Client
}

func newFaceSim(host string, client *rpc.Client) iFaceSim {
	return &_FaceSim{
		host:   host,
		Client: client,
	}
}

func (f _FaceSim) Eval(ctx context.Context, req *utility.FaceSimReq) (resp utility.FaceSimResp, err error) {
	err = f.CallWithJson(xlog.FromContextSafe(ctx), &resp, f.host+"/v1/face/sim", req)
	return
}

//--------------------------------------------------//
// ocr-idcard

type _EvalIDcardReq struct {
	Data struct {
		URI string `json:"uri"`
	} `json:"data"`
}

type _EvalIDcardResp struct {
	Code    int               `json:"code"`
	Message string            `json:"message"`
	Result  _EvalIDcardResult `json:"result"`
}

type _EvalIDcardResult struct {
	Name     string `json:"name"`
	People   string `json:"people"`
	Sex      string `json:"sex"`
	Address  string `json:"address"`
	IdNumber string `json:"id_number"`
}

type iDCardOcr interface {
	Eval(context.Context, *_EvalIDcardReq) (_EvalIDcardResp, error)
}

type _IdCard struct {
	host string
	*rpc.Client
}

func newIdCardOcr(host string, client *rpc.Client) iDCardOcr {

	return &_IdCard{
		host:   host,
		Client: client,
	}
}

func (d _IdCard) Eval(ctx context.Context, req *_EvalIDcardReq) (resp _EvalIDcardResp, err error) {
	err = d.CallWithJson(xlog.FromContextSafe(ctx), &resp, d.host+"/v1/eval/ocr-idcard", req)
	return
}

//------------------------------------------//
// official idcard database

type _IdDatabaeResp struct {
	SN       string `json:"sn"`
	Name     string `json:"name"`
	Sex      string `json:"sex"`
	Nation   string `json:"nation"`
	Birthday string `json:"birthday"`
	Address  string `json:"address"`
	Face     string `json:"face"`
}

type iDdatabase interface {
	Eval(context.Context, string) (_IdDatabaeResp, error)
}

type _IdDatabase struct {
	host    string
	timeout time.Duration
	*Client
}

func newIdDatabase(host string, client *Client, timeout time.Duration) iDdatabase {
	return &_IdDatabase{
		host:    host,
		Client:  client,
		timeout: timeout,
	}
}

func (db _IdDatabase) Eval(ctx context.Context, id string) (resp _IdDatabaeResp, err error) {
	xl := xlog.FromContextSafe(ctx)
	_, err = db.Client.Send("GET", db.host+"/rpc/customer/"+id+"/check", "application/json", db.timeout, nil)
	if err != nil { //第一次check会提交任务，若已经入库则迅速返回，否则返回418错误并尝试往后台数据库中拉信息，最多需要30s拉到
		xl.Errorf("check id info failed: %v, sleep 15s....", err)
		time.Sleep(15 * time.Second)
	}
	ret, err := db.Client.Send("GET", db.host+"/rpc/customer/"+id+"?detail", "application/json", db.timeout, nil)
	if err != nil {
		xl.Errorf("here1: %v", err)
		return
	}
	if len(ret) == 0 {
		err = httputil.NewError(http.StatusInternalServerError, "no valid data obtained from server")
	}
	err = json.Unmarshal(ret, &resp)
	return
}
