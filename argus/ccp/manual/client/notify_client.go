package client

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strings"

	"github.com/qiniu/rpc.v3"
	xlog "github.com/qiniu/xlog.v1"

	MODEL "qiniu.com/argus/cap/model"
	"qiniu.com/argus/ccp/conf"
	"qiniu.com/argus/ccp/manual/dao"
	"qiniu.com/argus/ccp/manual/enums"
	"qiniu.com/argus/ccp/manual/model"
	"qiniu.com/argus/ccp/manual/saver"
)

type ICcpNotify interface {
	CallBackCapResult(
		context.Context,
		string,
		error,
	) error
}

var _ ICcpNotify = _CcpNotify{}

type _CcpNotify struct {
	*conf.BatchEntryProcessorConf
	*model.CAPConfig
	SetDao        dao.ISetDAO
	BatchEntryDao dao.IBatchEntryDAO

	bucketSaver saver.IBucketSaver
}

func NewCapNotify(ctx context.Context,
	bconf *conf.BatchEntryProcessorConf,
	config *model.CAPConfig,
	setDao *dao.ISetDAO,
	batchDao *dao.IBatchEntryDAO,
	bucketSaver *saver.IBucketSaver,
) ICcpNotify {
	return _CcpNotify{
		BatchEntryProcessorConf: bconf,
		CAPConfig:               config,
		SetDao:                  *setDao,
		BatchEntryDao:           *batchDao,
		bucketSaver:             *bucketSaver,
	}
}

func (notify _CcpNotify) callBackFailResult(
	ctx context.Context,
	jId string,
	errInfo error,
) error {
	var (
		xl   = xlog.FromContextSafe(ctx)
		resp = model.NotifyToBCPResponse{}
	)

	xl.Info("begin call callBackFailResult")
	resp.Error = errInfo.Error()

	batchEntryInMgo, err := notify.BatchEntryDao.QueryByImageSetID(ctx, jId)
	if err != nil {
		xl.Errorf("notify.SetDao.QueryByID %s error: %#v", jId, err.Error())

		resp.Error = resp.Error + " " + err.Error()
	}
	// 发送结果给BCP
	setInMgo, err := notify.SetDao.QueryByID(ctx, batchEntryInMgo.SetId)
	if err != nil {
		xl.Errorf("notify.SetDao.QueryByID error: %#v", err.Error())
		resp.Error = resp.Error + " " + err.Error()
	}

	xl.Infof("notify resp: %#v", resp)
	return notify.postLastResult(ctx, setInMgo.NotifyURL, &resp)
}

//only for image
func (notify _CcpNotify) CallBackCapResult(
	ctx context.Context,
	jId string,
	errInfo error,
) error {
	//直接返回错误信息给ccp
	if errInfo != nil {
		return notify.callBackFailResult(ctx, jId, errInfo)
	}

	var (
		xl   = xlog.FromContextSafe(ctx)
		resp = model.NotifyToBCPResponse{}
	)
	batchEntryInMgo, err := notify.BatchEntryDao.QueryByImageSetID(ctx, jId)
	if err != nil {
		xl.Errorf("notify.SetDao.QueryByID %s error: %#v", jId, err.Error())
		return err
	}
	batchEntryStatus := enums.BatchEntryJobStatusSuccess

	//Query Cap Result
	marker := ""
	buf := bytes.NewBuffer(make([]byte, 0))
	bufLine := 0
	for {
		xl.Infof("marker val: %#v", marker)
		tasks, err := notify.queryTasksResult(ctx, jId, marker)
		if err != nil {
			xl.Errorf("notify.queryTasksResult err, %v", err)
			break
		}
		if len(tasks) <= 0 {
			break
		}

		xl.Infof("from cap task len: %#v", len(tasks))
		for _, task := range tasks {
			buf.WriteString(task.URI)
			buf.WriteString("\t")
			manualResp, err := json.Marshal(model.FromCAPResult(task))
			if err != nil {
				xl.Errorf("json.Marshal error: %#v", err.Error())
			}
			buf.WriteString(string(manualResp))
			buf.WriteString("\n")
			bufLine++

			//目前每1000000条记录存到一个文件
			if bufLine >= notify.BatchEntryProcessorConf.MaxFileLine {
				err = notify.bucketSaver.SaveResult(ctx, batchEntryInMgo.SetId, string(enums.MimeTypeImage), buf, bufLine)
				if err != nil {
					xl.Errorf("notify.saveCapResult error: %#v", err.Error())
					batchEntryStatus = enums.BatchEntryJobStatusFailed
					resp.Error = err.Error()
				}
				buf.Reset()
				bufLine = 0
			}
			// update marker
			marker = task.TaskID
		}
	}

	//剩余的CAP结果 -> 存入bucket
	if bufLine > 0 {
		err = notify.bucketSaver.SaveResult(ctx, batchEntryInMgo.SetId, string(enums.MimeTypeImage), buf, bufLine)
		if err != nil {
			xl.Errorf("notify.saveCapResult error: %#v", err.Error())
			batchEntryStatus = enums.BatchEntryJobStatusFailed
			resp.Error = err.Error()
		}
	}

	if batchEntryStatus != enums.BatchEntryJobStatusSuccess {

		//更新BatchEntry DB
		err = notify.BatchEntryDao.UpdateStatus(ctx, batchEntryInMgo.SetId, batchEntryStatus)
		if err != nil {
			xl.Errorf("notify.BatchEntryDao.UpdateStatus error: %#v", err.Error())
			resp.Error = err.Error()
		}
	}

	// 发送结果给BCP
	setInMgo, err := notify.SetDao.QueryByID(ctx, batchEntryInMgo.SetId)
	if err != nil {
		xl.Errorf("notify.SetDao.QueryByID error: %#v", err.Error())
		resp.Error = err.Error()
	}
	if resp.Error == "" {
		resp.Uid = setInMgo.Saver.UID
		resp.Bucket = setInMgo.Saver.Bucket
		resp.Keys = setInMgo.ResultFiles
	}

	xl.Infof("notify to ccp-manager resp: %#v", resp)
	return notify.postLastResult(ctx, setInMgo.NotifyURL, &resp)
}

//======================================================================================
func (notify _CcpNotify) queryTasksResult(ctx context.Context, jobID string,
	marker string) ([]*MODEL.TaskResult, error) {

	xl := xlog.FromContextSafe(ctx)

	req := MODEL.GetResultReq{
		CmdArgs: make([]string, 1),
		Marker:  marker,
		Limit:   500, //TODO:改成可配置的
	}
	req.CmdArgs[0] = jobID
	msg, err := json.Marshal(req)
	if err != nil {
		xl.Errorf("EndJob json.Marshal err, %v", err)
		return nil, err
	}
	xl.Info(string(msg))

	url := getTaskResultsUrl(notify.CAPConfig.Host, jobID)
	xl.Infof("queryTasksResult url: %#v", url)

	rpcClient := rpc.Client{Client: &http.Client{}}
	resp, err := rpcClient.DoRequestWith64Header(ctx, "GET", url,
		"application/json", bytes.NewReader(msg), int64(len(msg)), map[string]string{})
	if err != nil {
		xl.Errorf("GetRecords err, %v", err)
		return nil, err
	}

	defer resp.Body.Close()
	// 成功
	if resp.StatusCode/100 == 2 {
		xl.Info("GetRecords Success")

		buf := make([]byte, 1024*1024)
		n, err := resp.Body.Read(buf)
		if err != nil && err != io.EOF {
			xl.Errorf("GetRecords err, %v", err)
			return nil, err
		}
		taskstrs := strings.Split(string(buf[:n]), "\n")
		xl.Infof("GetRecords len, %d", len(taskstrs))

		var taskArrs []*MODEL.TaskResult
		for _, str := range taskstrs {
			if str != "" {
				itemstrs := strings.Split(str, "\t")
				if len(itemstrs) >= 3 {
					capResult := MODEL.TaskResult{
						TaskID: itemstrs[0],
						URI:    convQiniuUrl(itemstrs[1]),
					}

					var labels []MODEL.LabelInfo
					err := json.Unmarshal([]byte(itemstrs[2]), &labels)
					if err == nil {
						capResult.Labels = append(capResult.Labels, labels...)
					}

					taskArrs = append(taskArrs, &capResult)
				}
			}

		}
		return taskArrs, nil
	}

	xl.Errorf("GetRecords err, StatusCode = %d", resp.StatusCode)
	return nil, fmt.Errorf("GetRecords err, %d", resp.StatusCode)
}

func getTaskResultsUrl(host, jId string) string {
	return fmt.Sprintf("%s/v1/cap/job/%s/results", host, jId)
}

//===========================================================================
//将人审结果发给ccp-manager
func (notify _CcpNotify) postLastResult(ctx context.Context,
	notifyURL string,
	req *model.NotifyToBCPResponse,
) error {
	var (
		xl = xlog.FromContextSafe(ctx)
	)

	msg, err := json.Marshal(req)
	if err != nil {
		xl.Errorf("PostJobFinished json.Marshal err, ", err)
		return err
	}

	reader := bytes.NewReader(msg)
	rlen := len(msg)
	headerMap := map[string]string{}

	rpcClient := rpc.Client{Client: &http.Client{}}

	for i := 0; i < 5; i++ {
		resp, err := rpcClient.DoRequestWith64Header(ctx, "POST", notifyURL,
			"application/json", reader, int64(rlen), headerMap)
		if err != nil {
			xl.Errorf("PostMsgManual err: %#v", err)
			return err
		}
		if resp.StatusCode/100 == 2 {
			//成功将结果发送给ccp-manager
			xl.Infof("PostMsgManual resp success: %#v, ", resp)
			break
		}

		defer resp.Body.Close()
	}

	return err
}

//==============================================================================
func convQiniuUrl(qUrl string) string {
	u, err := url.Parse(qUrl)
	if err == nil && u.Scheme == "qiniu" {
		return fmt.Sprintf("%s://%s%s", u.Scheme, u.Host, u.Path)
	}
	return qUrl
}
