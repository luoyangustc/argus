package kmqcli

import (
	"encoding/json"
	"fmt"
	"io/ioutil"
	"net/http"
	"net/url"

	"github.com/qiniu/rpc.v1/lb.v2.1"
	"github.com/qiniu/xlog.v1"
	"qbox.us/digest_auth"
	"qiniupkg.com/x/rpc.v7"
)

type Config struct {
	AccessKey string   `json:"access_key"`
	SecretKey string   `json:"secret_key"`
	Hosts     []string `json:"hosts"`     // 目标服务地址列表，不能为空
	TryTimes  uint32   `json:"try_times"` // 对于某个请求 client 或者 failover 各自最大尝试次数，TryTimes = 重试次数 + 1
}

type Client struct {
	*Config
	lbClient *lb.Client
}

func New(cfg *Config) *Client {
	lbCfg := lb.Config{TryTimes: cfg.TryTimes, Hosts: cfg.Hosts}
	lbClient := lb.New(&lbCfg, digest_auth.NewTransport(cfg.AccessKey, cfg.SecretKey, http.DefaultTransport))
	return &Client{Config: cfg, lbClient: lbClient}
}

type queueInfo struct {
	Uid       uint32 `json:"uid"`
	Name      string `json:"name"`
	Retention int    `json:"retention"`
}

func dumpResp(resp *http.Response, xl *xlog.Logger) (res []byte, err error) {
	res, err = ioutil.ReadAll(resp.Body)
	if err != nil {
		xl.Error("dumpResp.ioutil.ReadAll", err)
		return
	}
	return res, nil
}

func (client *Client) CreateQueue(uid uint32, name string, retention int, xl *xlog.Logger) (statusCode int, err error) {
	url := fmt.Sprintf("/queues")
	queue := queueInfo{Uid: uid, Name: name, Retention: retention}
	resp, err := client.lbClient.PostWithJson(xl, url, queue)
	if err != nil {
		xl.Error("CreateQueue.LbClient.PostWithJson", err)
		return 500, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != 200 {
		return resp.StatusCode, rpc.ResponseError(resp)
	}
	return resp.StatusCode, err
}

func (client *Client) DeleteQueue(uid uint32, name string, xl *xlog.Logger) (statusCode int, err error) {
	url := fmt.Sprintf("/queues/delete")
	queue := queueInfo{Uid: uid, Name: name}
	resp, err := client.lbClient.PostWithJson(xl, url, queue)
	if err != nil {
		xl.Error("DeleteQueue.LbClient.PostWithJson", err)
		return 500, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != 200 {
		return resp.StatusCode, rpc.ResponseError(resp)
	}
	return resp.StatusCode, err
}

type KmqInfo struct {
	Name      string `json:"name"`
	Retention int    `json:"retention"`
}

func (client *Client) GetQueuesByUid(uid uint32, xl *xlog.Logger) (statusCode int, queues []KmqInfo, err error) {
	url := fmt.Sprintf("/queues?uid=%d", uid)
	resp, err := client.lbClient.Get(xl, url)
	if err != nil {
		xl.Error("GetQueuesByUid.LbClient.Get", err)
		return 500, queues, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != 200 {
		return resp.StatusCode, queues, rpc.ResponseError(resp)
	}
	res, err := dumpResp(resp, xl)
	if err != nil {
		xl.Error("GetQueuesByUid.dumpResp", err)
		return 500, queues, err
	}
	type KmqInfos struct {
		Items []KmqInfo `json:"items"`
	}
	var kmqInfos KmqInfos
	err = json.Unmarshal(res, &kmqInfos)
	if err != nil {
		xl.Error("GetQueuesByUid.json.Unmarshal", err)
		return 500, queues, err
	}
	return resp.StatusCode, kmqInfos.Items, err
}

func (client *Client) GetQueueRetention(uid uint32, name string, xl *xlog.Logger) (statusCode int, retention int, err error) {
	url := fmt.Sprintf("/queues/%s?uid=%d", name, uid)
	resp, err := client.lbClient.Get(xl, url)
	if err != nil {
		xl.Error("GetQueueRetention.LbClient.Get", err)
		return 500, retention, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != 200 {
		return resp.StatusCode, retention, rpc.ResponseError(resp)
	}
	res, err := dumpResp(resp, xl)
	if err != nil {
		xl.Error("GetQueueRetention.dumpResp", err)
		return 500, retention, err
	}
	type GetQueueRetentionRet struct {
		Retention int `json:"retention"`
	}
	var reten GetQueueRetentionRet
	err = json.Unmarshal(res, &reten)
	if err != nil {
		xl.Error("GetQueueRetention.json.Unmarshal", err)
		return 500, retention, err
	}
	return resp.StatusCode, reten.Retention, err
}

func (client *Client) GetQueuePartitions(uid uint32, name string, xl *xlog.Logger) (statusCode int, partitions []int32, err error) {
	url := fmt.Sprintf("/queues/%s/admin/partitions?uid=%d", name, uid)
	resp, err := client.lbClient.Get(xl, url)
	if err != nil {
		xl.Error("GetQueuePartitions.LbClient.Get", err)
		return 500, partitions, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != 200 {
		return resp.StatusCode, partitions, rpc.ResponseError(resp)
	}
	res, err := dumpResp(resp, xl)
	if err != nil {
		xl.Error("GetQueuePartitions.dumpResp", err)
		return 500, partitions, err
	}
	type GetQueuePartitionsRet struct {
		Partitions []int32 `json:"partitions"`
	}
	var parts GetQueuePartitionsRet
	err = json.Unmarshal(res, &parts)
	if err != nil {
		xl.Error("GetQueuePartitions.json.Unmarshal", err)
		return 500, partitions, err
	}
	return resp.StatusCode, parts.Partitions, err
}

func (client *Client) ChangeQueueRetention(uid uint32, name string, retention int, xl *xlog.Logger) (statusCode int, err error) {
	url := fmt.Sprintf("/queues/%s", name)
	queue := queueInfo{Uid: uid, Retention: retention}
	resp, err := client.lbClient.PostWithJson(xl, url, queue)
	if err != nil {
		xl.Error("ChangeQueueRetention.LbClient.PostWithJson", err)
		return 500, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != 200 {
		return resp.StatusCode, rpc.ResponseError(resp)
	}
	return resp.StatusCode, err
}

func (client *Client) ProduceMessages(name string, msgs []string, xl *xlog.Logger) (statusCode int, err error) {
	url := fmt.Sprintf("/queues/%s/produce", name)
	type Msgs struct {
		Msgs []string `json:"msgs"`
	}
	data := Msgs{Msgs: msgs}
	resp, err := client.lbClient.PostWithJson(xl, url, data)
	if err != nil {
		xl.Error("ProduceMessages.LbClient.PostWithJson", err)
		return 500, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != 200 {
		return resp.StatusCode, rpc.ResponseError(resp)
	}
	return resp.StatusCode, err
}

func (client *Client) ProduceMessagesAdmin(uid uint32, name string, msgs []string, xl *xlog.Logger) (statusCode int, err error) {
	url := fmt.Sprintf("/queues/%s/admin/produce", name)
	type AdminMsgs struct {
		Msgs []string `json:"msgs"`
		Uid  uint32   `json:"uid"`
	}
	data := AdminMsgs{Msgs: msgs, Uid: uid}
	resp, err := client.lbClient.PostWithJson(xl, url, data)
	if err != nil {
		xl.Error("ProduceMessagesAdmin.LbClient.PostWithJson", err)
		return 500, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != 200 {
		return resp.StatusCode, rpc.ResponseError(resp)
	}
	return resp.StatusCode, err
}

type consumeMessagesRet struct {
	Msgs     []string `json:"msgs"`
	Position string   `json:"position"`
}

func (client *Client) ConsumeMessages(name string, position string, limit int, xl *xlog.Logger) (statusCode int, msgs []string, nextPosition string, err error) {
	params := url.Values{}
	params.Set("position", position)
	params.Set("limit", fmt.Sprintf("%d", limit))
	url := fmt.Sprintf("/queues/%s/consume?%s", name, params.Encode())
	resp, err := client.lbClient.Get(xl, url)
	if err != nil {
		xl.Error("ConsumeMessages.LbClient.Get", err)
		return 500, msgs, nextPosition, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != 200 {
		return resp.StatusCode, msgs, nextPosition, rpc.ResponseError(resp)
	}
	res, err := dumpResp(resp, xl)
	if err != nil {
		xl.Error("ConsumeMessages.dumpResp", err)
		return 500, msgs, nextPosition, err
	}
	var m consumeMessagesRet
	err = json.Unmarshal(res, &m)
	if err != nil {
		xl.Error("ConsumeMessages.json.Unmarshal", err)
		return 500, msgs, nextPosition, err
	}
	return resp.StatusCode, m.Msgs, m.Position, err
}

func (client *Client) ConsumeMessagesAdmin(uid uint32, name string, position string, limit int, xl *xlog.Logger) (statusCode int, msgs []string, nextPosition string, err error) {
	params := url.Values{}
	params.Set("position", position)
	params.Set("limit", fmt.Sprintf("%d", limit))
	params.Set("uid", fmt.Sprintf("%d", uid))
	url := fmt.Sprintf("/queues/%s/admin/consume?%s", name, params.Encode())
	resp, err := client.lbClient.Get(xl, url)
	if err != nil {
		xl.Error("ConsumeMessagesAdmin.LbClient.Get", err)
		return 500, msgs, nextPosition, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != 200 {
		return resp.StatusCode, msgs, nextPosition, rpc.ResponseError(resp)
	}
	res, err := dumpResp(resp, xl)
	if err != nil {
		xl.Error("ConsumeMessagesAdmin.dumpResp", err)
		return 500, msgs, nextPosition, err
	}
	var m consumeMessagesRet
	err = json.Unmarshal(res, &m)
	if err != nil {
		xl.Error("ConsumeMessagesAdmin.json.Unmarshal", err)
		return 500, msgs, nextPosition, err
	}
	return resp.StatusCode, m.Msgs, m.Position, err
}

func (client *Client) ConsumeMessagesByPartitonAdmin(uid uint32, name string, position string, limit int, partition int32, xl *xlog.Logger) (statusCode int, msgs []string, nextPosition string, err error) {
	params := url.Values{}
	params.Set("position", position)
	params.Set("limit", fmt.Sprintf("%d", limit))
	params.Set("uid", fmt.Sprintf("%d", uid))
	params.Set("partition", fmt.Sprintf("%d", partition))
	url := fmt.Sprintf("/queues/%s/admin/consume/partition?%s", name, params.Encode())
	resp, err := client.lbClient.Get(xl, url)
	if err != nil {
		xl.Error("ConsumeMessagesByPartitonAdmin.LbClient.Get", err)
		return 500, msgs, nextPosition, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != 200 {
		return resp.StatusCode, msgs, nextPosition, rpc.ResponseError(resp)
	}
	res, err := dumpResp(resp, xl)
	if err != nil {
		xl.Error("ConsumeMessagesByPartitonAdmin.dumpResp", err)
		return 500, msgs, nextPosition, err
	}
	var m consumeMessagesRet
	err = json.Unmarshal(res, &m)
	if err != nil {
		xl.Error("ConsumeMessagesByPartitonAdmin.json.Unmarshal", err)
		return 500, msgs, nextPosition, err
	}
	return resp.StatusCode, m.Msgs, m.Position, err
}
