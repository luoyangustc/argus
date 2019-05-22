package proxy_hook

import (
	"encoding/json"
	"net/http"

	"qiniu.com/argus/fop/pulp_ufop/proxy/cmd"
)

const auditCmd string = cmd.Audit

func extraTerror(data []byte) map[string]interface{} {
	var o map[string]interface{}
	if err := json.Unmarshal(data, &o); err != nil {
		panic("bug:parser terror error:" + err.Error())
	}

	code, ok := o["code"]
	if !ok {
		panic("bug: no 'code' field")
	}

	codeNumber, ok := code.(float64)
	if !ok {
		panic("bug: 'code' field's value is not number")
	}

	if codeNumber != 0 {
		return map[string]interface{}{
			"message": o["message"],
		}
	}

	fileList, ok := o["fileList"]
	if !ok {
		panic("bug: no fieldList field")
	}

	sliceResult, ok := fileList.([]interface{})
	if !ok {
		panic("bug: 'fileList' field's value is not slice")
	}

	if len(sliceResult) <= 0 {
		panic("bug: 'fileList' field's value length is 0")
	}

	first, ok := sliceResult[0].(map[string]interface{})
	if !ok {
		panic("api result 'fileList' fist value is not object")
	}

	delete(first, "name")

	return first
}

func audit(req *http.Request, data []byte) (http.Header, []byte) {
	cmds := req.URL.Query()["cmd"]
	if len(cmds) <= 0 || cmds[0] != auditCmd {
		return nil, data
	}

	var o map[string]string

	if err := json.Unmarshal(data, &o); err != nil {
		panic("bugging:unmarshal data " + err.Error())
	}

	h, pulp := pulp0(req, []byte(o[cmd.Pulp]))

	var retObject map[string]interface{}
	if err := json.Unmarshal(pulp, &retObject); err != nil {
		panic("bugging:unmarshal pulp " + err.Error())
	}

	retObject["terror"] = extraTerror([]byte(o[cmd.Audit]))

	r, err := json.Marshal(retObject)
	if err != nil {
		panic("bug: serials error:" + err.Error())
	}

	return h, r
}

func init() {
	registerAfterRequest(afterRequestHook(audit))
}
