package proxy_server

import (
	"encoding/json"
	"errors"
	"io/ioutil"
	"log"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"sync/atomic"
	"testing"

	"qiniu.com/argus/fop/pulp_ufop/proxy/config"
	"qiniu.com/argus/fop/pulp_ufop/proxy/resource"
)

type response struct {
	Code    int    `json:"code"`
	Message string `json:"message"`
}

type mockSucResource struct{}

func (mr mockSucResource) Upload(filepath, key string) (string, error) {
	log.Println("mock upload suc ", filepath, key)
	return "", nil
}

func (mr mockSucResource) Delete(url string) error {
	log.Println("mock delete suc")
	return nil
}

type mockFailedResource struct{}

func (mf mockFailedResource) Upload(filepath, key string) (string, error) {
	return "", errors.New("mock failed upload")
}

func (mf mockFailedResource) Delete(url string) error {
	return errors.New("mock fail delete")
}

type mockClient struct{}

func (client mockClient) PostForm(url string,
	params url.Values) ([]byte, error) {
	resp, err := http.PostForm(url, params)
	if err != nil {
		return nil, err
	}

	data, err := ioutil.ReadAll(resp.Body)
	resp.Body.Close()

	if err != nil {
		return nil, err
	}

	if resp.StatusCode != http.StatusOK {
		return nil, errors.New(string(data))
	}

	return data, nil
}

func (client mockClient) Get(string, url.Values) ([]byte, error) {
	return nil, nil
}

func (client mockClient) Set(uid, utype uint32) {
	return
}

var (
	pulpReviewRet = `{
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
	}`
	pulpNoReviewRet = `{
			"code":0,
			"message":"pulp success",
			"pulp": {
				"reviewCount":0,
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
	}`

	pulpRequest int32 = 0
)

func createHttpServer(r proxy_resource.Resource,
	cmds []proxy_config.Cmd, maxConcurrent uint32) *httptest.Server {
	handler := http.NewServeMux()
	handler.Handle("/health", http.HandlerFunc(createHealthHandler()))

	handler.Handle("/handler", http.HandlerFunc(createHandler(mockClient{}, r,
		cmds, maxConcurrent)))

	handler.Handle("/image.gif", http.HandlerFunc(func(rw http.ResponseWriter,
		r *http.Request) {
		log.Println("request image")
		rw.Header().Add("Content-Type", "image/gif")
		rw.Write([]byte("GIF89a!,L;"))
	}))

	handler.Handle("/pulp", http.HandlerFunc(func(rw http.ResponseWriter,
		r *http.Request) {
		rw.Header().Add("Content-Type", "application/json")
		if atomic.LoadInt32(&pulpRequest)%2 == 0 {
			rw.Write([]byte(pulpReviewRet))
		} else {
			rw.Write([]byte(pulpNoReviewRet))
		}

		atomic.AddInt32(&pulpRequest, 1)
	}))

	ts := httptest.NewServer(handler)
	for i, c := range cmds {
		cmds[i].Url = ts.URL + "/" + c.Name
	}
	return ts
}

func initConfig() (ret *proxy_config.Config) {
	ret, _ = proxy_config.LoadFromData([]byte(`
	{
		"port": 9090,
		"max_concurrent":2,
		"bucket": {
			"ak":"ak",
			"sk":"sk",
			"name":"name",
			"domain":"domain"
		},
		"cmds": [
			{
				"name":"pulp",
				"url":"pulp.url"
			},
			{
				"name":"facex",
				"url":"facex.url"
			}
		]
	}`))

	return
}

func assertResponse(httpCode, code int, errorMessage string, t *testing.T,
	resp *http.Response) (data []byte) {
	if httpCode != resp.StatusCode {
		t.Error("http code is invalide, expected:", httpCode, ", actual:",
			resp.StatusCode)
	}

	data, err := ioutil.ReadAll(resp.Body)
	if err != nil {
		t.Fatal("read response data error", err)
	}
	defer resp.Body.Close()

	ret := map[string]interface{}{}
	err = json.Unmarshal(data, &ret)
	if err != nil {
		t.Fatal("unmarshal data error", err)
	}

	t.Log(ret)
	actualCode := ret["code"]
	switch actualCode.(type) {
	case float64:
		if float64(code) != actualCode.(float64) {
			t.Error("code is invalide, expected:", code, ", actual:",
				actualCode)
		}
	case int:
		if code != actualCode.(int) {
			t.Error("code is invalide, expected:", code, ", actual:",
				actualCode)
		}
	default:
		t.Error("unknow code:", actualCode, " type")

	}

	if errorMessage == "" {
		return
	}

	if errorMessage != ret["message"].(string) {
		t.Error("error message is invalide, expected:", errorMessage,
			", actual:", ret["message"].(string))
	}
	return
}

func TestGroup(t *testing.T) {
	config := initConfig()
	ts := createHttpServer(mockFailedResource{}, config.Cmds,
		config.MaxConcurrent)
	defer ts.Close()

	t.Run("health", func(t *testing.T) {
		req, err := http.NewRequest("GET", ts.URL+"/health", nil)
		if err != nil {
			t.Fatal(err)
		}
		req.Header.Set("X-Qiniu-Uid", "MTM0OTg1Nw==")
		resp, err := http.DefaultClient.Do(req)
		if err != nil {
			t.Fatal(err)
		}
		assertResponse(http.StatusOK, 0, "hi", t, resp)
	})
	t.Run("no-url", func(t *testing.T) {
		req, err := http.NewRequest("GET", ts.URL+"/handler", nil)
		if err != nil {
			t.Fatal(err)
		}
		req.Header.Set("X-Qiniu-Uid", "MTM0OTg1Nw==")
		resp, err := http.DefaultClient.Do(req)
		if err != nil {
			t.Fatal(err)
		}
		assertResponse(http.StatusBadRequest, http.StatusBadRequest, "no url",
			t, resp)
	})
	t.Run("no-cmd", func(t *testing.T) {

		req, err := http.NewRequest("GET", ts.URL+"/handler?url=abc", nil)
		if err != nil {
			t.Fatal(err)
		}
		req.Header.Set("X-Qiniu-Uid", "MTM0OTg1Nw==")
		resp, err := http.DefaultClient.Do(req)
		if err != nil {
			t.Fatal(err)
		}
		assertResponse(http.StatusBadRequest, http.StatusBadRequest, "no cmd",
			t, resp)
	})
	t.Run("unsupport-cmd", func(t *testing.T) {
		req, err := http.NewRequest("GET", ts.URL+"/handler?url=abc&cmd=no-cmd", nil)
		if err != nil {
			t.Fatal(err)
		}
		req.Header.Set("X-Qiniu-Uid", "MTM0OTg1Nw==")
		resp, err := http.DefaultClient.Do(req)
		if err != nil {
			t.Fatal(err)
		}
		assertResponse(http.StatusBadRequest, http.StatusBadRequest,
			"unrecognized cmd:no-cmd",
			t, resp)
	})
	t.Run("downlod-failed", func(t *testing.T) {
		req, err := http.NewRequest("GET", ts.URL+"/handler?url=http://test.avapulp.jpg&cmd=pulp", nil)
		if err != nil {
			t.Fatal(err)
		}
		req.Header.Set("X-Qiniu-Uid", "MTM0OTg1Nw==")
		resp, err := http.DefaultClient.Do(req)
		if err != nil {
			t.Fatal(err)
		}
		assertResponse(400, 400, "", t, resp)
	})
	t.Run("upload-failed", func(t *testing.T) {

		req, err := http.NewRequest("GET", ts.URL+"/handler?url="+ts.URL+"/image.gif&cmd=pulp", nil)
		if err != nil {
			t.Fatal(err)
		}
		req.Header.Set("X-Qiniu-Uid", "MTM0OTg1Nw==")
		resp, err := http.DefaultClient.Do(req)
		if err != nil {
			t.Fatal(err)
		}
		assertResponse(500, 100, "upload file error:mock failed upload", t, resp)
	})
	t.Run("post no cmd", func(t *testing.T) {
		req, err := http.NewRequest("POST", ts.URL+"/handler", strings.NewReader("GIF89a!,L;"))
		if err != nil {
			t.Fatal(err)
		}
		req.Header.Set("X-Qiniu-Uid", "MTM0OTg1Nw==")
		req.Header.Set("Content-Type", "image/gif")
		resp, err := http.DefaultClient.Do(req)
		if err != nil {
			t.Fatal(err)
		}
		assertResponse(http.StatusBadRequest, http.StatusBadRequest, "no cmd",
			t, resp)
	})
}

func TestSuc(t *testing.T) {
	config := initConfig()
	ts := createHttpServer(mockSucResource{}, config.Cmds, config.MaxConcurrent)
	defer ts.Close()
	pulpRequest = 0
	t.Run("suc reivew", func(t *testing.T) {

		req, err := http.NewRequest("GET", ts.URL+"/handler?url="+ts.URL+"/image.gif&cmd=pulp", nil)
		if err != nil {
			t.Fatal(err)
		}
		req.Header.Set("X-Qiniu-Uid", "MTM0OTg1Nw==")
		resp, err := http.DefaultClient.Do(req)
		if err != nil {
			t.Fatal(err)
		}
		assertResponse(http.StatusOK, 0, "pulp success", t, resp)
		reviewHeader := resp.Header.Get("X-Origin-A")
		if reviewHeader != "PULP_Depend,1" {
			t.Error("invalide review count:" + reviewHeader)
		}
	})
	t.Run("suc no reivew", func(t *testing.T) {

		req, err := http.NewRequest("GET", ts.URL+"/handler?url="+ts.URL+"/image.gif&cmd=pulp", nil)
		if err != nil {
			t.Fatal(err)
		}
		req.Header.Set("X-Qiniu-Uid", "MTM0OTg1Nw==")
		resp, err := http.DefaultClient.Do(req)

		if err != nil {
			t.Fatal(err)
		}
		assertResponse(http.StatusOK, 0, "pulp success", t, resp)
		reviewHeader := resp.Header.Get("X-Origin-A")
		if reviewHeader != "PULP_Certain,1" {
			t.Error("invalide review count:" + reviewHeader)
		}
	})
	t.Run("test body", func(t *testing.T) {

		req, err := http.NewRequest("POST", ts.URL+"/handler?cmd=pulp", strings.NewReader("GIF89a!,L;"))
		if err != nil {
			t.Fatal(err)
		}
		req.Header.Set("X-Qiniu-Uid", "MTM0OTg1Nw==")
		req.Header.Set("Content-Type", "image/gif")
		resp, err := http.DefaultClient.Do(req)
		if err != nil {
			t.Fatal(err)
		}
		assertResponse(http.StatusOK, 0, "pulp success", t, resp)
	})
}
