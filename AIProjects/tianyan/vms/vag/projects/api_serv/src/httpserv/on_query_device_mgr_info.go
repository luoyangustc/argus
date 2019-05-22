package httpserv

import (
	"encoding/json"
	"io/ioutil"
	"net/http"
	"time"

	httputil "qiniupkg.com/http/httputil.v2"
	log "qiniupkg.com/x/log.v7"
)

type queryDeviceMgrInfoResp struct {
	Code int    `json:"code"`
	Msg  string `json:"msg"`
	Data struct {
		Subdevices []subdeviceMgrInfo `json:"sub_devices"`
	} `json:"data"`
}

func (s *Service) onQueryDeviceMgrInfo(resp http.ResponseWriter, req *http.Request) {
	respInfo := new(queryDeviceMgrInfoResp)
	respInfo.Msg = " Success"
	respInfo.Code = respCodeSuccess

	for {
		httpClient := &http.Client{
			Timeout: 5 * time.Second,
		}

		reqURL := "http://" + s.Config.MgtServHost + req.RequestURI
		resp, err := httpClient.Get(reqURL)
		if err != nil {
			respInfo.Msg = "connect mgt serv error"
			respInfo.Code = respCodeServerErr
			break
		}

		defer resp.Body.Close()
		body, err := ioutil.ReadAll(resp.Body)
		if err != nil {
			respInfo.Msg = "query device mgr info fail"
			respInfo.Code = respCodeServerErr
			break
		}

		if err := json.Unmarshal(body, respInfo); err != nil {
			respInfo.Msg = "device mgr info unmarshal fail"
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

	log.Info("onQueryDeviceMgrInfo |", req.RemoteAddr, "|", req.RequestURI, "|", respInfo.Code, "|", respInfo.Msg)
}
