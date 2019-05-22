package httpserv

import (
	"encoding/json"
	"io/ioutil"
	"net/http"
	"time"

	httputil "qiniupkg.com/http/httputil.v2"
	log "qiniupkg.com/x/log.v7"
)

func (s *Service) onInternalReq(resp http.ResponseWriter, req *http.Request) {
	msg := " Success"
	code := respCodeSuccess

	for {
		httpClient := &http.Client{
			Timeout: 5 * time.Second,
		}

		reqURL := "http://" + s.Config.MgtServHost + req.RequestURI
		tmpResp, err := httpClient.Get(reqURL)
		if err != nil {
			msg = "connect mgt serv error"
			code = respCodeServerErr
			break
		}

		defer tmpResp.Body.Close()
		body, err := ioutil.ReadAll(tmpResp.Body)
		if err != nil {
			msg = "read response msg fail"
			code = respCodeServerErr
			break
		}
		httputil.ReplyWith(resp, 200, "application/json", body)
		break
	}

	if code != respCodeSuccess {
		respInfo := new(commResp)
		respInfo.Code = code
		respInfo.Msg = msg
		data, err := json.MarshalIndent(respInfo, " ", " ")
		if err == nil {
			httputil.ReplyWith(resp, 200, "application/json", data)
		} else {
			httputil.Error(resp, err)
		}
	}

	log.Info("onInternalReq |", req.RemoteAddr, "|", req.RequestURI, "|", code, "|", msg)
}
