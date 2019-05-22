package proxy_hook

import (
	"encoding/json"
	"net/http"
	"testing"
)

func TestBadData(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Error("failed", r)
		}
	}()
	req, _ := http.NewRequest("GET", "http://abc?url=abc&cmd="+pulpCmd, nil)
	pulp(req, nil, pulpCmd)
}
func TestBadData1(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Error("failed", r)
		}
	}()
	req, _ := http.NewRequest("GET", "http://abc?url=abc&cmd="+pulpCmd, nil)
	pulp(req, []byte("["), pulpCmd)
}

func TestNoPulp(t *testing.T) {
	req, _ := http.NewRequest("GET", "http://abc?url=abc&cmd="+pulpCmd, nil)
	h, _ := pulp(req, []byte(`{
			"code":0,
			"message":"pulp success"
	}`), pulpCmd)

	if h != nil {
		t.Error("failed header")
	}
}

func TestNoList(t *testing.T) {
	defer func() {
		r := recover()
		if r == nil {
			t.Error("failed", r)
		}
		if r.(string) != "api result no 'fileList' field" {
			t.Error("failed", r)
		}
	}()
	req, _ := http.NewRequest("GET", "http://abc?url=abc&cmd="+pulpCmd, nil)
	pulp(req, []byte(`{
			"code":0,
			"message":"pulp success",
			"pulp": {
				"reviewCount":1,
				"statistic": [1, 0,0]
			}
	}`), pulpCmd)
}
func TestNotList(t *testing.T) {
	defer func() {
		r := recover()
		if r == nil {
			t.Error("failed", r)
		}
		if r.(string) != "api result 'fileList' field's value is not slice" {
			t.Error("failed", r)
		}
	}()
	req, _ := http.NewRequest("GET", "http://abc?url=abc&cmd="+pulpCmd, nil)
	pulp(req, []byte(`{
			"code":0,
			"message":"pulp success",
			"pulp": {
				"reviewCount":1,
				"statistic": [1, 0,0],
				"fileList":0
			}
	}`), pulpCmd)
}

func TestEmptyList(t *testing.T) {
	defer func() {
		r := recover()
		if r == nil {
			t.Error("failed", r)
		}
		if r.(string) != "api result 'fileList' field's value length is 0" {
			t.Error("failed", r)
		}
	}()
	req, _ := http.NewRequest("GET", "http://abc?url=abc&cmd="+pulpCmd, nil)
	pulp(req, []byte(`{
			"code":0,
			"message":"pulp success",
			"pulp": {
				"reviewCount":1,
				"statistic": [1, 0,0],
				"fileList": [
				]
			}
	}`), pulpCmd)
}

func TestNoResult(t *testing.T) {
	defer func() {
		r := recover()
		if r == nil {
			t.Error("failed", r)
		}
		if r.(string) != "api result is not mapppppppping" {
			t.Error("failed", r)
		}
	}()
	req, _ := http.NewRequest("GET", "http://abc?url=abc&cmd="+pulpCmd, nil)
	pulp(req, []byte(`{
			"code":0,
			"message":"pulp success",
			"pulp": {
				"reviewCount":1,
				"statistic": [1, 0,0],
				"fileList": [
				{
				}
				]
			}
	}`), pulpCmd)
}

func TestSuc(t *testing.T) {
	req, _ := http.NewRequest("GET", "http://abc?url=abc&cmd="+pulpCmd, nil)
	h, data := pulp(req, []byte(`{
			"code":0,
			"message":"pulp success",
			"pulp": {
				"reviewCount":1,
				"statistic": [1, 0,0],
				"fileList": [
				{
					"result": {
						"rate":0.99,
						"label":0,
						"name":"http://image.net/abc.jpeg",
						"review":true
					}
				}
				]
			}
	}`), pulpCmd)

	if h.Get("X-Origin-A") != "PULP_Depend,1" {
		t.Error("head error")
	}

	ret := map[string]interface{}{}

	if err := json.Unmarshal(data, &ret); err != nil {
		t.Error("unmarshal error", err)
	}

	pulpRet := ret[pulpCmd]
	if pulpRet == nil {
		t.Error("no ", pulpCmd)
	}

	pulpMap := pulpRet.(map[string]interface{})
	if len(pulpMap) != 3 {
		t.Error("length error")
	}
	if pulpMap["rate"].(float64) != 0.99 {
		t.Error("rate error")
	}

	if pulpMap["label"].(float64) != 0 {
		t.Error("label error")
	}

	if pulpMap["review"].(bool) != true {
		t.Error("review error")
	}
}
