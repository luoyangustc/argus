package httpserv

import (
	"encoding/json"
	"io/ioutil"
	"net/http"
	"time"

	httputil "qiniupkg.com/http/httputil.v2"
	log "qiniupkg.com/x/log.v7"
)

type getSessionServerResp struct {
	Code int    `json:"code"`
	Msg  string `json:"msg"`
	Data struct {
		IP    string `json:"ip"`
		Port  int    `json:"port"`
		Token string `json:"token"`
	} `json:"data"`
}

func (s *Service) onGetSessionServer(resp http.ResponseWriter, req *http.Request) {
	respInfo := new(getSessionServerResp)

	for {
		httpClient := &http.Client{
			Timeout: 5 * time.Second,
		}

		reqURL := "http://" + s.Config.StatusServHost + "/get_session_server"

		deviceID := req.FormValue("device_id")
		if deviceID != "" {
			reqURL += "?device_id=" + deviceID
		}

		resp, err := httpClient.Get(reqURL)
		if err != nil {
			respInfo.Msg = "connect status serv error"
			respInfo.Code = respCodeServerErr
			break
		}

		defer resp.Body.Close()
		body, err := ioutil.ReadAll(resp.Body)
		if err != nil {
			respInfo.Msg = "get getSessionServerResp fail"
			respInfo.Code = respCodeServerErr
			break
		}

		if err := json.Unmarshal(body, respInfo); err != nil {
			respInfo.Msg = "get_session_server unmarshal fail"
			respInfo.Code = respCodeServerErr
			break
		}

		break
	}

	data, err := json.MarshalIndent(respInfo, " ", " ")
	if err == nil {
		httputil.ReplyWith(resp, 200, "application/json", data)
	} else {
		httputil.Error(resp, err)
	}

	log.Info("onGetSessionServer |", req.RemoteAddr, "|", req.RequestURI, "|", respInfo.Code, "|", respInfo.Msg)
}
