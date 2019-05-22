package proxy_client

import (
	"encoding/base64"
	"errors"
	"io/ioutil"
	"log"
	"net/http"
	"net/url"
	"strconv"

	"fmt"
	"qiniu.com/auth/qboxmac.v1"
	"qiniupkg.com/x/rpc.v7"
	"time"
)

type client struct {
	ak, sk  string
	uid     uint32
	utype   uint32
	timeout time.Duration
}

func CreateAuthClient(ak, sk string, uid, utype uint32, timeout time.Duration) Client {
	return &client{ak: ak, sk: sk, uid: uid, utype: utype, timeout: timeout}
}

type transport struct {
	mac       qboxmac.Mac
	uid       uint32
	Transport http.RoundTripper
}

func (t *transport) RoundTrip(req *http.Request) (resp *http.Response, err error) {
	sign, err := qboxmac.SignRequest(t.mac.SecretKey, req)
	if err != nil {
		return
	}

	auth := "QBox " + t.mac.AccessKey + ":" + base64.URLEncoding.EncodeToString(sign)
	req.Header.Set("Authorization", auth)
	marsAuth := "uid=" + strconv.FormatUint(uint64(t.uid), 10)
	req.Header.Set("MarsAuth", marsAuth)
	return t.Transport.RoundTrip(req)
}

type qiniuStubTransport struct {
	uid   uint32
	utype uint32
	http.RoundTripper
}

func (t qiniuStubTransport) RoundTrip(req *http.Request) (*http.Response, error) {
	req.Header.Set(
		"Authorization",
		fmt.Sprintf("QiniuStub uid=%d&ut=%d", t.uid, t.utype))
	return t.RoundTripper.RoundTrip(req)
}

func NewQiniuStubRPCClient(uid, utype uint32, timeout time.Duration) *rpc.Client {
	return &rpc.Client{
		Client: &http.Client{
			Timeout: timeout,
			Transport: qiniuStubTransport{
				uid:          uid,
				utype:        utype,
				RoundTripper: http.DefaultTransport,
			},
		},
	}
}

func (client *client) doRequest(url string,
	params url.Values,
	req func(string, url.Values, rpc.Client) (*http.Response, error)) ([]byte,
	error) {
	log.Println("call ", url, params)

	rpcClient := rpc.Client{
		Client: &http.Client{
			Timeout:   client.timeout,
			Transport: &transport{qboxmac.Mac{client.ak, []byte(client.sk)}, 0, http.DefaultTransport},
		},
	}

	//rpcClient := NewQiniuStubRPCClient(client.uid, client.utype, client.timeout)

	resp, err := req(url, params, rpcClient)
	if err != nil {
		return nil, err
	}

	data, err := ioutil.ReadAll(resp.Body)
	resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, errors.New(string(data))
	}

	return data, nil
}

func (client *client) Set(uid, utype uint32) {
	client.uid = uid
	client.utype = utype
}

func (client *client) Get(u string, params url.Values) ([]byte, error) {
	return client.doRequest(u, params,
		func(u string, params url.Values, client rpc.Client) (*http.Response,
			error) {
			if len(params) > 0 {
				u += "?" + params.Encode()
			}
			return client.Get(u)
		})
}

func (client *client) PostForm(u string, params url.Values) ([]byte, error) {
	return client.doRequest(u, params,
		func(u string, params url.Values, client rpc.Client) (*http.Response,
			error) {
			return client.DoRequestWithForm(nil, http.MethodPost, u, params)
		})
}
