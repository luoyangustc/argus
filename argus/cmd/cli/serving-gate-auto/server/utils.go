package server

import (
	"encoding/json"
	"net/http"
	"strings"

	"github.com/qiniu/xlog.v1"
)

const MaxBodySize = 16 * 1024 * 1024

var xl = xlog.NewWith("main")

func ce(err error) {
	if err != nil {
		xl.Panicln(err)
	}
}

func copyHeader(dst, src http.Header) {
	for k, vv := range src {
		for _, v := range vv {
			dst.Add(k, v)
		}
	}
}

func getAppName(path string) string {
	if strings.HasPrefix(path, "/v1/eval/") {
		path = path[len("/v1/eval/"):]
		if r := strings.Index(path, "/"); r > 0 {
			return "ava-" + path[:r]
		}
		return "ava-" + path
	}
	return ""
}

const dataURIPrefix = "data:application/octet-stream;base64,"

func (s *server) fixBodyBuf(buf []byte) []byte {
	var r interface{}
	err := json.Unmarshal(buf, &r)
	if err != nil {
		return buf
	}
	processData := func(data map[string]interface{}) {
		if uri, ok := data["uri"].(string); ok {
			uri2, err := s.fixUri(uri)
			if err != nil {
				xl.Warn("fix uri error", err, uri)
			} else {
				data["uri"] = uri2
			}
		}
	}
	if r, ok := r.(map[string]interface{}); ok {
		if dataArr, ok := r["data"].([]interface{}); ok {
			for _, dataArrItem := range dataArr {
				if data, ok := dataArrItem.(map[string]interface{}); ok {
					processData(data)
				}
			}
		}
		if data, ok := r["data"].(map[string]interface{}); ok {
			processData(data)
		}
	}
	buf2, err := json.Marshal(r)
	ce(err)
	return buf2
}
