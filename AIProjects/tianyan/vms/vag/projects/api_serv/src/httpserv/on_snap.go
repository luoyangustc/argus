package httpserv

import (
	"encoding/json"
	"io/ioutil"
	"net/http"
	"time"

	"qiniupkg.com/http/httputil.v2"
	"qiniupkg.com/x/log.v7"
)

type snapResp struct {
	Code int    `json:"code"`
	Msg  string `json:"msg"`
	Data struct {
		PicURL string `json:"pic_url"`
	} `json:"data"`
}

func (s *Service) onSnap(resp http.ResponseWriter, req *http.Request) {
	respInfo := new(snapResp)
	preview := "0"
	for {

		deviceID := req.FormValue("device_id")
		channelID := req.FormValue("channel_id")
		preview = req.FormValue("preview")

		if deviceID == "" || channelID == "" {
			respInfo.Msg = "param device_id or channel_id is null"
			respInfo.Code = respCodeReqParamErr
			break
		}

		if preview == "" {
			preview = "0"
		}

		sessionHTTPHost := ""
		respInfo.Code, respInfo.Msg, sessionHTTPHost = s.getDevSessionHTTPHost(deviceID)
		if respInfo.Code != respCodeSuccess {
			break
		}

		reqURL := "http://" + sessionHTTPHost + "/snap?device_id=" + deviceID + "&channel_id=" + channelID

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

	if preview == "1" && respInfo.Code == respCodeSuccess && respInfo.Data.PicURL != "" {
		resp.Header().Set("Location", respInfo.Data.PicURL)
		httputil.ReplyWith(resp, 302, "", []byte(""))
	} else {
		data, err := json.MarshalIndent(respInfo, " ", " ")
		if err == nil {
			//data = bytes.Replace(data, []byte("\\u0026"), []byte("&"), -1)
			httputil.ReplyWith(resp, 200, "application/json", data)
		} else {
			httputil.Error(resp, err)
		}
	}

	log.Info("onSnap |", req.RemoteAddr, "|", req.RequestURI, "|", respInfo.Code, "|", respInfo.Msg)
}
