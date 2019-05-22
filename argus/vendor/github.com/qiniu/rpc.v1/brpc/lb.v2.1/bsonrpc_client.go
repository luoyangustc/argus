package lb

import (
	"bytes"
	"io/ioutil"
	"net/http"

	"github.com/qiniu/rpc.v1"
	"github.com/qiniu/rpc.v1/lb.v2.1"
	"gopkg.in/mgo.v2/bson"
)

// --------------------------------------------------------------------

func bsonDecode(v interface{}, resp *http.Response) error {

	b, err := ioutil.ReadAll(resp.Body)
	if err != nil {
		return err
	}
	return bson.Unmarshal(b, v)
}

func callRet(l rpc.Logger, ret interface{}, resp *http.Response) (err error) {

	defer resp.Body.Close()

	if resp.StatusCode/100 == 2 {
		if ret != nil && resp.ContentLength != 0 {
			err = bsonDecode(ret, resp)
			if err != nil {
				return
			}
		}
		if resp.StatusCode == 200 {
			return nil
		}
	}
	return rpc.ResponseError(resp)
}

// --------------------------------------------------------------------

type Config struct {
	lb.Config
}

type Client struct {
	cli *lb.Client
}

func New(cfg *Config, tr http.RoundTripper) *Client {
	return &Client{lb.New(&cfg.Config, tr)}
}

func NewWithFailover(client, failover *Config, clientTr, failoverTr http.RoundTripper, shouldFailover func(int, error) bool) *Client {
	return &Client{lb.NewWithFailover(&client.Config, &failover.Config, clientTr, failoverTr, shouldFailover)}
}

// --------------------------------------------------------------------

func (r *Client) PostWithBson(l rpc.Logger, url1 string, data interface{}) (resp *http.Response, err error) {

	msg, err := bson.Marshal(data)
	if err != nil {
		return
	}
	return r.cli.PostWith(l, url1, "application/bson", bytes.NewReader(msg), len(msg))
}

func (r *Client) PostWithForm(l rpc.Logger, url1 string, param map[string][]string) (resp *http.Response, err error) {
	return r.cli.PostWithForm(l, url1, param)
}

func (r *Client) CallWithForm(l rpc.Logger, ret interface{}, url1 string, param map[string][]string) (err error) {

	resp, err := r.cli.PostWithForm(l, url1, param)
	if err != nil {
		return err
	}
	return callRet(l, ret, resp)
}

func (r *Client) CallWithBson(l rpc.Logger, ret interface{}, url1 string, param interface{}) (err error) {

	resp, err := r.PostWithBson(l, url1, param)
	if err != nil {
		return err
	}
	return callRet(l, ret, resp)
}

func (r *Client) Call(l rpc.Logger, ret interface{}, url1 string) (err error) {

	resp, err := r.cli.PostWith(l, url1, "application/x-www-form-urlencoded", nil, 0)
	if err != nil {
		return err
	}
	return callRet(l, ret, resp)
}

// --------------------------------------------------------------------
