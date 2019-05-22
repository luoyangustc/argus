package httpserv

import (
	"encoding/json"
	"fmt"
	"io/ioutil"
	"net/http"
	"strconv"
	"time"

	"qiniupkg.com/http/httputil.v2"
	"qiniupkg.com/x/log.v7"
)

type liveResp struct {
	Code int    `json:"code"`
	Msg  string `json:"msg"`
	Data struct {
		PlayURL string `json:"play_url"`
	} `json:"data"`
}

func (s *Service) onLive(resp http.ResponseWriter, req *http.Request) {
	respInfo := new(liveResp)

	for {

		deviceID := req.FormValue("device_id")
		channelID := req.FormValue("channel_id")
		streamID := req.FormValue("stream_id")
		playType := req.FormValue("type")

		if deviceID == "" || channelID == "" {
			respInfo.Msg = "param device_id or channel_id is null"
			respInfo.Code = respCodeReqParamErr
			break
		}

		if streamID == "" {
			streamID = "0"
		}

		if playType == "" {
			playType = "rtmp"
		}

		sessionHTTPHost := ""
		respInfo.Code, respInfo.Msg, sessionHTTPHost = s.getDevSessionHTTPHost(deviceID)
		log.Info("getDevSessionHTTPHost |", req.RemoteAddr, "|", sessionHTTPHost, "|", respInfo.Code, "|", respInfo.Msg)
		if respInfo.Code != respCodeSuccess {
			break
		}

		reqURL := "http://" + sessionHTTPHost +
			"/live?device_id=" + deviceID + "&channel_id=" + channelID +
			"&stream_id=" + streamID + "&type=" + playType

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
			respInfo.Msg = "live resp unmarshal fail"
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

	log.Info("onLive |", req.RemoteAddr, "|", req.RequestURI, "|", respInfo.Code, "|", respInfo.Msg)
}

type LiveResult struct {
	ChannelID int    `json:"channel_id"`
	PlayURL   string `json:"play_url"`
}

type testLiveResp struct {
	Code int    `json:"code"`
	Msg  string `json:"msg"`
	Data struct {
		Results []LiveResult `json:"result_list"`
	} `json:"data"`
}

func (s *Service) onTestLive(resp http.ResponseWriter, req *http.Request) {
	respInfo := new(testLiveResp)
	for {
		deviceID := req.FormValue("device_id")
		startChID := req.FormValue("start_ch_id")
		endChID := req.FormValue("end_ch_id")
		streamID := req.FormValue("stream_id")
		playType := req.FormValue("type")

		if deviceID == "" || startChID == "" || endChID == "" {
			respInfo.Msg = "param device_id or channel_id is null"
			respInfo.Code = respCodeReqParamErr
			break
		}

		nStartID, err := strconv.Atoi(startChID)
		if err != nil {
			respInfo.Msg = "start channel id incorrect"
			respInfo.Code = respCodeServerErr
			break
		}

		nEndID, err := strconv.Atoi(endChID)
		if err != nil {
			respInfo.Msg = "end channel id incorrect"
			respInfo.Code = respCodeServerErr
			break
		}

		if nStartID > nEndID {
			respInfo.Msg = "start_ch_id > end_ch_id"
			respInfo.Code = respCodeServerErr
			break
		}

		results := []LiveResult{}
		for i := nStartID; i <= nEndID; i++ {
			channelID := fmt.Sprintf("%d", i)
			_, _, playURL := s.openLive(deviceID, channelID, streamID, playType)
			var r LiveResult
			r.ChannelID = i
			r.PlayURL = playURL
			results = append(results, r)
		}
		respInfo.Data.Results = results
		respInfo.Msg = "success"
		respInfo.Code = respCodeSuccess
		break
	}

	data, err := json.MarshalIndent(respInfo, " ", " ")
	if err == nil {
		httputil.ReplyWith(resp, 200, "application/json", data)
	} else {
		httputil.Error(resp, err)
	}

	log.Info("testLiveResp |", req.RemoteAddr, "|", req.RequestURI, "|", respInfo.Code, "|", respInfo.Msg)
}

func (s *Service) openLive(deviceID string, channelID string, streamID string, playType string) (code int, msg string, playURL string) {

	for {
		if deviceID == "" || channelID == "" {
			msg = "param device_id or channel_id is null"
			code = respCodeReqParamErr
			break
		}

		if streamID == "" {
			streamID = "0"
		}

		if playType == "" {
			playType = "rtmp"
		}

		sessionHTTPHost := ""
		code, msg, sessionHTTPHost = s.getDevSessionHTTPHost(deviceID)
		if code != respCodeSuccess {
			break
		}

		reqURL := "http://" + sessionHTTPHost +
			"/live?device_id=" + deviceID + "&channel_id=" + channelID +
			"&stream_id=" + streamID + "&type=" + playType

		httpClient := &http.Client{
			Timeout: 5 * time.Second,
		}

		resp, err := httpClient.Get(reqURL)
		if err != nil {
			msg = "connect session serv error"
			code = respCodeServerErr
			break
		}

		defer resp.Body.Close()
		body, err := ioutil.ReadAll(resp.Body)
		if err != nil {
			msg = "get resp fail"
			code = respCodeServerErr
			break
		}

		respInfo := new(liveResp)
		if err := json.Unmarshal(body, respInfo); err != nil {
			msg = "live resp unmarshal fail"
			code = respCodeServerErr
			break
		}

		msg = respInfo.Msg
		code = respInfo.Code
		playURL = respInfo.Data.PlayURL

		break
	}

	return
}
