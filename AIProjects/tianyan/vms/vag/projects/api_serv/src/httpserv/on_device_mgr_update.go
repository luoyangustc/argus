package httpserv

import (
	"encoding/json"
	"io/ioutil"
	"net/http"
	"time"

	"qiniupkg.com/http/httputil.v2"
	"qiniupkg.com/x/log.v7"
)

func (s *Service) onDeviceMgrUpdate(resp http.ResponseWriter, req *http.Request) {
	respInfo := new(commResp)
	for {

		deviceID := req.FormValue("device_id")
		channelID := req.FormValue("channel_id")
		mgrType := req.FormValue("mgr_type")

		if deviceID == "" || channelID == "" || mgrType == "" {
			respInfo.Msg = "param device_id or channel_id is null"
			respInfo.Code = respCodeReqParamErr
			break
		}

		sessionHTTPHost := ""
		respInfo.Code, respInfo.Msg, sessionHTTPHost = s.getDevSessionHTTPHost(deviceID)
		if respInfo.Code != respCodeSuccess {
			break
		}

		reqURL := "http://" + sessionHTTPHost +
			"/device_mgr_update?device_id=" + deviceID +
			"&channel_id=" + channelID +
			"&mgr_type=" + mgrType

		httpClient := &http.Client{
			Timeout: 5 * time.Second,
		}

		resp, err := httpClient.Get(reqURL)
		if err != nil {
			respInfo.Msg = "connect session serv error"
			respInfo.Code = respCodeServerErr
			break
		}

		defer resp.Body.Close()
		body, err := ioutil.ReadAll(resp.Body)
		if err != nil {
			respInfo.Msg = "get resp fail"
			respInfo.Code = respCodeServerErr
			break
		}

		if err := json.Unmarshal(body, respInfo); err != nil {
			respInfo.Msg = "resp unmarshal fail"
			respInfo.Code = respCodeServerErr
			break
		}

		break
	}

	data, err := json.MarshalIndent(respInfo, " ", " ")
	if err == nil {
		//data = bytes.Replace(data, []byte("\\u0026"), []byte("&"), -1)
		httputil.ReplyWith(resp, 200, "application/json", data)
	} else {
		httputil.Error(resp, err)
	}

	log.Info("onDeviceMgrUpdate |", req.RemoteAddr, "|", req.RequestURI, "|", respInfo.Code, "|", respInfo.Msg)
}
