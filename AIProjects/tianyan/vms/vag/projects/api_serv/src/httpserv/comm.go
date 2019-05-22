package httpserv

import (
	"encoding/json"
	"fmt"
	"io/ioutil"
	"net/http"
	"strconv"
	"strings"
	"time"

	httputil "qiniupkg.com/http/httputil.v2"
)

const (
	respCodeSuccess        = 0
	respCodeServerErr      = 10001
	respCodeReqNosupport   = 10002
	respCodeReqParamErr    = 10003
	respCodeDeviceOffline  = 10004
	respCodeChannelOffline = 10005
)

type deviceSessionStatus struct {
	DeviceID      string `json:"device_id"`
	Status        string `json:"status"`
	TimeStamp     string `json:"timestamp"`
	SessionHost   string `json:"session_serv_addr"`
	ChannelNum    int    `json:"channel_num"`
	ChannelStatus string `json:"channel_status"`
}

type devicesStatus struct {
	DeviceNum int                   `json:"device_num"`
	Devices   []deviceSessionStatus `json:"devices"`
}

type commResp struct {
	Code int      `json:"code"`
	Msg  string   `json:"msg"`
	Data struct{} `json:"data"`
}

func (s *Service) getDevSessionHTTPHost(deviceID string) (code int, msg string, sessionHTTPHost string) {
	for {
		httpClient := &http.Client{
			Timeout: 5 * time.Second,
		}

		reqURL := "http://" + s.Config.StatusServHost + "/query_device_status?device_id=" + deviceID
		resp, err := httpClient.Get(reqURL)
		if err != nil {
			msg = "connect status serv error"
			code = respCodeServerErr
			break
		}

		defer resp.Body.Close()
		body, err := ioutil.ReadAll(resp.Body)
		if err != nil {
			msg = "get device status fail"
			code = respCodeServerErr
			break
		}

		var dat devicesStatus
		if err := json.Unmarshal(body, &dat); err != nil {
			msg = "device status unmarshal fail"
			code = respCodeServerErr
			break
		}

		if dat.DeviceNum != 1 || len(dat.Devices) != 1 {
			msg = "device status parse fail"
			code = respCodeServerErr
			break
		}

		if dat.Devices[0].Status != "online" {
			msg = "device offline"
			code = respCodeDeviceOffline
			break
		}

		host := strings.Split(dat.Devices[0].SessionHost, ":")
		if len(host) != 2 {
			msg = "device session host incorrect"
			code = respCodeServerErr
			break
		}

		port, err := strconv.Atoi(host[1])
		if err != nil {
			msg = "device session host port incorrect"
			code = respCodeServerErr
			break
		}

		httpPort := port + 10
		sessionHTTPHost = fmt.Sprintf("%s:%d", host[0], httpPort)
		msg = "success"
		code = respCodeSuccess
		break
	}

	return
}

func (s *Service) onLocation(resp http.ResponseWriter, req *http.Request) {
	respData := []byte("<html><title>result</title><body>QN API server!</body></html>")
	httputil.ReplyWith(resp, 200, "text/html", respData)
	return
}
