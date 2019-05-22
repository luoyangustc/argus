package proxy_hook

import (
	"encoding/json"
	"net/http"

	"qiniu.com/argus/fop/pulp_ufop/proxy/cmd"
)

const pulpCmd string = cmd.Pulp

func pulp(req *http.Request, data []byte, cmd string) (http.Header, []byte) {
	cmds := req.URL.Query()["cmd"]
	if len(cmds) <= 0 || cmds[0] != cmd {
		return nil, data
	}

	return pulp0(req, data)
}

func pulp0(req *http.Request, data []byte) (http.Header, []byte) {
	ret := map[string]interface{}{}
	if err := json.Unmarshal(data, &ret); err != nil {
		panic("api result is error, not json data")
	}

	if ret[pulpCmd] == nil {
		return nil, data
	}

	pulp := ret[pulpCmd].(map[string]interface{})
	reviewCount := pulp["reviewCount"]

	v := "PULP_Certain,1"
	if reviewCount != nil && reviewCount.(float64) > 0 {
		v = "PULP_Depend,1"
	}

	delete(ret, pulpCmd)

	fileList := pulp["fileList"]
	if fileList == nil {
		panic("api result no 'fileList' field")
	}

	sliceResult, ok := fileList.([]interface{})
	if !ok {
		panic("api result 'fileList' field's value is not slice")
	}

	if len(sliceResult) <= 0 {
		panic("api result 'fileList' field's value length is 0")
	}

	first, ok := sliceResult[0].(map[string]interface{})
	if !ok {
		panic("api result 'fileList' fist value is not object")
	}

	result, ok := first["result"].(map[string]interface{})
	if !ok {
		panic("api result is not mapppppppping")
	}

	delete(result, "name")

	ret[pulpCmd] = result
	newData, err := json.Marshal(ret)

	if err != nil {
		panic("api result no result")
	}

	return http.Header{
		"X-Origin-A": {v},
	}, newData
}
func init() {
	registerAfterRequest(afterRequestHook(
		func(req *http.Request, data []byte) (http.Header, []byte) {
			return pulp(req, data, pulpCmd)
		}))
}
