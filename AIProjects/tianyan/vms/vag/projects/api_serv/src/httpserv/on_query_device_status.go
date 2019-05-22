package httpserv

import (
	"encoding/json"
	"io/ioutil"
	"net/http"
	"time"

	httputil "qiniupkg.com/http/httputil.v2"
	log "qiniupkg.com/x/log.v7"
)

type queryDeviceStatusResp struct {
	Code int    `json:"code"`
	Msg  string `json:"msg"`
	Data struct {
		devicesStatus
	} `json:"data"`
}

func (s *Service) onQueryDeviceStatus(resp http.ResponseWriter, req *http.Request) {
	respInfo := new(queryDeviceStatusResp)
	respInfo.Msg = "success"
	respInfo.Code = respCodeSuccess
	for {
		httpClient := &http.Client{
			Timeout: 5 * time.Second,
		}

		deviceID := req.FormValue("device_id")
		if deviceID == "" {
			respInfo.Msg = "param device_id is null"
			respInfo.Code = respCodeReqParamErr
			break
		}

		reqURL := "http://" + s.Config.StatusServHost + "/query_device_status?device_id=" + deviceID
		resp, err := httpClient.Get(reqURL)
		if err != nil {
			respInfo.Msg = "connect status serv error"
			respInfo.Code = respCodeServerErr
			break
		}

		defer resp.Body.Close()
		body, err := ioutil.ReadAll(resp.Body)
		if err != nil {
			respInfo.Msg = "get device status fail"
			respInfo.Code = respCodeServerErr
			break
		}

		var dat devicesStatus
		if err := json.Unmarshal(body, &dat); err != nil {
			respInfo.Msg = "device status unmarshal fail"
			respInfo.Code = respCodeServerErr
			break
		}
		respInfo.Data.devicesStatus = dat

		break
	}

	data, err := json.MarshalIndent(respInfo, " ", " ")
	if err == nil {
		httputil.ReplyWith(resp, 200, "application/json", data)
	} else {
		httputil.Error(resp, err)
	}

	log.Info("onQueryDeviceStatus |", req.RemoteAddr, "|", req.RequestURI, "|", respInfo.Code, "|", respInfo.Msg)
}
