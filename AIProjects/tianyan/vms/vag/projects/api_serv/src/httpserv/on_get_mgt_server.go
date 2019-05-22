package httpserv

import (
	"encoding/json"
	"net/http"
	"strconv"
	"strings"

	httputil "qiniupkg.com/http/httputil.v2"
	log "qiniupkg.com/x/log.v7"
)

type getMgtServerResp struct {
	Code int    `json:"code"`
	Msg  string `json:"msg"`
	Data struct {
		IP   string `json:"ip"`
		Port int    `json:"port"`
	} `json:"data"`
}

func (s *Service) onGetMgtServer(resp http.ResponseWriter, req *http.Request) {
	respInfo := new(getMgtServerResp)
	respInfo.Msg = " Success"
	respInfo.Code = respCodeSuccess

	for {
		host := strings.Split(s.Config.MgtServHost, ":")
		if len(host) == 1 {
			respInfo.Data.IP = host[0]
			respInfo.Data.Port = 80
		} else if len(host) == 2 {
			port, err := strconv.Atoi(host[1])
			if err != nil {
				respInfo.Msg = "mgt host port incorrect"
				respInfo.Code = respCodeServerErr
				break
			}
			respInfo.Data.IP = host[0]
			respInfo.Data.Port = port
		} else {
			respInfo.Msg = "mgt host incorrect"
			respInfo.Code = respCodeServerErr
		}
		break
	}

	data, err := json.MarshalIndent(respInfo, " ", " ")
	if err == nil {
		httputil.ReplyWith(resp, 200, "application/json", data)
	} else {
		httputil.Error(resp, err)
	}

	log.Info("onGetMgtServer |", req.RemoteAddr, "|", req.RequestURI, "|", respInfo.Code, "|", respInfo.Msg)
}
