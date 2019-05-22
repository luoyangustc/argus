package httpserv

import (
	"encoding/json"
	"io/ioutil"
	"net/http"
	"time"

	httputil "qiniupkg.com/http/httputil.v2"
	log "qiniupkg.com/x/log.v7"
)

type subdeviceMgrInfo struct {
	ChannelID int `json:"channel_id"`
	DevType   int `json:"type"`
	DevAttr   struct {
		Name           string `json:"name"`
		IP             string `json:"ip"`
		DiscoveryProto string `json:"discovery_protocol"`
		Account        string `json:"account"`
		Password       string `json:"password"`
		Vendor         string `json:"vendor"`
	} `json:"attribute"`
}

type querySubdeviceMgrInfoResp struct {
	Code int    `json:"code"`
	Msg  string `json:"msg"`
	Data struct {
		subdeviceMgrInfo
	} `json:"data"`
}

func (s *Service) onQuerySubdeviceMgrInfo(resp http.ResponseWriter, req *http.Request) {
	respInfo := new(querySubdeviceMgrInfoResp)
	respInfo.Msg = " Success"
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
			respInfo.Msg = "query camera mgr info fail"
			respInfo.Code = respCodeServerErr
			break
		}

		if err := json.Unmarshal(body, respInfo); err != nil {
			respInfo.Msg = "camera mgr info unmarshal fail"
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

	log.Info("onQuerySubdeviceMgrInfo |", req.RemoteAddr, "|", req.RequestURI, "|", respInfo.Code, "|", respInfo.Msg)
}
