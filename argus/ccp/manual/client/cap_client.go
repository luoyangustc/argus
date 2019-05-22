package client

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"sort"
	"strings"

	xlog "github.com/qiniu/xlog.v1"

	"github.com/qiniu/rpc.v3"
	ENUMS "qiniu.com/argus/cap/enums"
	MODEL "qiniu.com/argus/cap/model"
	"qiniu.com/argus/ccp/manual/model"
)

type ICAPClient interface {
	NewJob(context.Context, ENUMS.MimeType, *model.SetModel) (string, error)
	PushBatchTask(context.Context, string, []*model.BatchTasksReq) error
	CheckJob(context.Context, string) (bool, error)
}

type _CAPClient struct {
	*model.CAPConfig
}

func NewCAPClient(config *model.CAPConfig) ICAPClient {
	return &_CAPClient{
		CAPConfig: config,
	}
}

func (c _CAPClient) NewJob(ctx context.Context, mimeType ENUMS.MimeType, setModel *model.SetModel) (string, error) {
	var (
		xl        = xlog.FromContextSafe(ctx)
		rpcClient = rpc.Client{Client: &http.Client{}}
	)

	xl.Infof("setModel.Image.Scenes: %d, %#v", len(setModel.Image.Scenes), setModel.Image.Scenes)
	req := MODEL.JobCreateReq{
		JobID:     fmt.Sprintf("%s_%s", setModel.SetId, mimeType),
		JobType:   setModel.Type,
		LabelMode: GetCapMode(ctx, setModel.Image.Scenes),
		MimeType:  mimeType,
		Uid:       setModel.UID,
	}
	xl.Infof("newJob params: %#v", req)

	msg, err := json.Marshal(req)
	if err != nil {
		xl.Errorf("NewJob json.Marshal err, %v", err)
		return "", err
	}
	xl.Infof("msg infof:%#v", string(msg))
	url := getNewJobUrl(c.CAPConfig.Host)
	xl.Info(string(url))

	resp, err := rpcClient.DoRequestWith64Header(
		ctx, "POST", url, "application/json",
		bytes.NewReader(msg), int64(len(msg)), map[string]string{})
	if err != nil {
		xl.Errorf("NewJob err, %v", err)
		return "", err
	}

	defer resp.Body.Close()
	// 成功
	if resp.StatusCode/100 == 2 {
		xl.Info("NewJob Success")
		return req.JobID, nil
	}

	xl.Errorf("NewJob err, StatusCode = %d", resp.StatusCode)
	return "", fmt.Errorf("NewJob err, %d", resp.StatusCode)
}

func (c _CAPClient) PushBatchTask(
	ctx context.Context,
	jobId string,
	tasks []*model.BatchTasksReq,
) error {
	var (
		xl = xlog.FromContextSafe(ctx)
	)

	req := MODEL.JobTaskReq{
		CmdArgs: []string{jobId},
	}

	for _, v := range tasks {
		req.Tasks = append(req.Tasks, struct {
			ID     string            `json:"id"`
			URI    string            `json:"uri"` // HTTP|QINIU
			Labels []MODEL.LabelInfo `json:"label"`
		}{
			ID:     v.ID,
			URI:    v.URI,
			Labels: v.Labels,
		})
	}

	msg, err := json.Marshal(req)
	if err != nil {
		xl.Errorf("PushTasks json.Marshal err, %v", err)
		return err
	}

	url := getPushBatchTaskUrl(c.CAPConfig.Host, jobId)

	rpcClient := rpc.Client{Client: &http.Client{}}
	resp, err := rpcClient.DoRequestWith64Header(ctx, "POST", url,
		"application/json", bytes.NewReader(msg), int64(len(msg)), map[string]string{})
	if err != nil {
		xl.Errorf("PushTasks err, %v", err)
		return err
	}

	defer resp.Body.Close()
	// 成功
	if resp.StatusCode/100 == 2 {
		xl.Info("PushTasks Success")
		return nil
	}

	xl.Errorf("PushTasks err, StatusCode = %d", resp.StatusCode)
	return fmt.Errorf("PushTasks err, %d", resp.StatusCode)
}

func (c _CAPClient) CheckJob(ctx context.Context, jobId string) (bool, error) {
	var (
		xl        = xlog.FromContextSafe(ctx)
		rpcClient = rpc.Client{Client: &http.Client{}}
	)

	req := MODEL.JobCheckResultReq{
		CmdArgs: make([]string, 1),
	}
	req.CmdArgs[0] = jobId

	msg, err := json.Marshal(req)
	if err != nil {
		xl.Errorf("NewJob json.Marshal err, %v", err)
		return false, err
	}

	url := getCheckJobUrl(c.CAPConfig.Host, jobId)

	resp, err := rpcClient.DoRequestWith64Header(
		ctx, "POST", url, "application/json",
		bytes.NewReader(msg), int64(len(msg)), map[string]string{})
	if err != nil {
		xl.Errorf("checkJob err, %v", err)
		return false, err
	}

	if resp.StatusCode/100 != 2 {
		xl.Infof("Job %s finished error and the resp.StatusCode is %#v", jobId, resp.StatusCode)
		return false, fmt.Errorf("checkJob err, %d", resp.StatusCode)
	}

	defer resp.Body.Close()
	buf := make([]byte, 1024*1024)
	n, err := resp.Body.Read(buf)
	if err != nil && err != io.EOF {
		xl.Errorf("GetRecords err, %v", err)
		return false, err
	}
	val := strings.Split(string(buf[:n]), "\n")
	var bFinish MODEL.JobCheckResultResp

	err = json.Unmarshal([]byte(val[0]), &bFinish)
	if err != nil {
		xl.Infof("json.Unmarshal error: %#v", err.Error())
		return false, err
	}

	if bFinish.Finish {
		xl.Infof("jobid : %s bFinish.Finish val: %#v", jobId, bFinish.Finish)
	}

	return bFinish.Finish, nil
}

//====================================================================//
func getNewJobUrl(host string) string {
	return fmt.Sprintf("%s/v1/cap/job", host)
}
func getPushBatchTaskUrl(host, jobId string) string {
	return fmt.Sprintf("%s/v1/cap/job/%s/tasks", host, jobId)
}
func getCheckJobUrl(host, jobId string) string {
	return fmt.Sprintf("%s/v1/cap/job/%s/check/result", host, jobId)
}

//=====================================================================
type lts struct {
	labelTypes []string
}

func (lts lts) Len() int {
	return len(lts.labelTypes)
}

func (lts lts) Less(i, j int) bool {
	return lts.labelTypes[i] < lts.labelTypes[j]
}

func (lts lts) Swap(i, j int) {
	lts.labelTypes[i], lts.labelTypes[j] = lts.labelTypes[j], lts.labelTypes[i]
}

func GetCapMode(ctx context.Context, labelTypes []string) string {

	// "pulp", "terror", "politician"
	lts := lts{
		labelTypes: labelTypes,
	}
	sort.Sort(lts)

	mode := "mode"
	for _, lt := range lts.labelTypes {
		mode += "_"
		mode += lt
	}

	return mode
}
