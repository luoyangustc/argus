package proxy_server

import (
	"encoding/base64"
	"fmt"
	"io/ioutil"
	"log"
	"net/http"
	"os"
	"strconv"
	"time"

	"qiniu.com/argus/fop/pulp_ufop/proxy/client"
	"qiniu.com/argus/fop/pulp_ufop/proxy/config"
	"qiniu.com/argus/fop/pulp_ufop/proxy/hook"
	"qiniu.com/argus/fop/pulp_ufop/proxy/message"
	"qiniu.com/argus/fop/pulp_ufop/proxy/request"
	"qiniu.com/argus/fop/pulp_ufop/proxy/resource"
	"qiniu.com/argus/fop/pulp_ufop/proxy/uuid"

	"github.com/qiniu/http/httputil.v1"
	"github.com/qiniu/xlog.v1"
)

func createKey(filepath string) (key string) {
	key = uuid.NewV1().String()
	return
}

func writeReponse(rw http.ResponseWriter, httpCode int, header http.Header,
	data []byte) {
	rwHeader := rw.Header()
	rwHeader.Set("Content-Type", "application/json; charset=utf-8")
	if len(header) != 0 {
		for k, vs := range header {
			for _, v := range vs {
				rwHeader.Add(k, v)
			}
		}
	}
	rw.WriteHeader(httpCode)
	rw.Write(data)
}

func saveData(data []byte) (string, int, int, error) {
	tmpfile, err := ioutil.TempFile("", "ufop-proxy")
	if err != nil {
		log.Println("create temp file error", err)
		return "", 500, 500, err
	}

	tmpfile.Write(data)
	tmpfile.Close()

	return tmpfile.Name(), 0, 0, nil
}

func downloadImage(url string) (string, int, int, error) {
	log.Println("download url:" + url)
	client := http.Client{
		Timeout: time.Duration(1 * time.Minute),
	}

	resp, err := client.Get(url)
	if err != nil {
		log.Println("download error", err)
		return "", 400, 400, err
	}

	data, err := ioutil.ReadAll(resp.Body)
	if err != nil {
		log.Println("read body error")
		return "", 400, 400, err
	}
	resp.Body.Close()

	return saveData(data)
}

func getArguments(req *http.Request) (urls []string,
	cmds []string,
	isUrl bool,
	uid uint32,
	err error) {

	xl := xlog.NewDummy()
	urls, cmds = req.URL.Query()["url"], req.URL.Query()["cmd"]
	if len(urls) != 0 {
		isUrl = true
	} else {
		d, _err := ioutil.ReadAll(req.Body)
		if _err == nil && len(d) != 0 {
			isUrl = false
			urls = []string{string(d)}
		}
		if _err != nil {
			err = _err
			xl.Errorf("read body error:", err, ",data:", string(d))
			return
		}
	}

	str := req.Header.Get("X-Qiniu-Uid")
	if str == "" {
		err = httputil.NewError(http.StatusBadRequest, "need params: uid")
		xl.Error("got X-Qiniu-Uid uid error")
		return
	}
	bs, err := base64.StdEncoding.DecodeString(str)
	if err != nil {
		xl.Errorf("decode X-Qiniu-Uid uid error:", err)
		err = httputil.NewError(
			http.StatusBadRequest,
			fmt.Sprintf("bad params: uid, %s", str),
		)
		return
	}

	_uid, err := strconv.ParseUint(string(bs), 10, 64)
	if err != nil {
		xl.Errorf("parse X-Qiniu-Uid uid error:", err)
		err = httputil.NewError(
			http.StatusBadRequest,
			fmt.Sprintf("bad params: uid, %s", string(bs)),
		)
		return
	}
	uid = uint32(_uid)
	return
}

func process(req *http.Request, client proxy_client.Client,
	resource proxy_resource.Resource,
	cmds []proxy_config.Cmd) (httpStatusCode int, data []byte) {
	urls, cmd, isUrl, uid, err := getArguments(req)

	if err != nil {
		return http.StatusBadRequest, proxy_message.CreateError(400,
			"image data error")
	}

	if len(urls) == 0 {
		return http.StatusBadRequest, proxy_message.CreateError(400, "no url")
	}

	if len(cmd) == 0 {
		return http.StatusBadRequest, proxy_message.CreateError(400, "no cmd")
	}

	var reqCmd *proxy_config.Cmd
	for _, c := range cmds {
		if c.Name == cmd[0] {
			reqCmd = &c
			break
		}
	}

	if reqCmd == nil {
		return http.StatusBadRequest, proxy_message.CreateError(400,
			"unrecognized cmd:"+cmd[0])
	}

	var (
		filepath string
		code     int
	)

	if isUrl {
		filepath, httpStatusCode, code, err = downloadImage(urls[0])
	} else {
		filepath, httpStatusCode, code, err = saveData([]byte(urls[0]))
	}

	if err != nil {
		return httpStatusCode, proxy_message.CreateError(code, err.Error())
	}

	defer os.Remove(filepath)

	key := createKey(urls[0])
	uploadUrl, err := resource.Upload(filepath, key)
	if err != nil {
		return http.StatusInternalServerError, proxy_message.CreateError(100,
			"upload file error:"+err.Error())
	}
	defer resource.Delete(key)
	client.Set(uid, 0)
	// newClient := proxy_client.CreateAuthClient(, "", uid, 0, 120*time.Second)
	data, err = request.Do(reqCmd.Name, reqCmd.Url, uploadUrl, client)
	if err != nil {
		return http.StatusInternalServerError, proxy_message.CreateError(100,
			"call api error:"+err.Error())
	}

	return http.StatusOK, data
}

func createHandler(client proxy_client.Client,
	resource proxy_resource.Resource,
	cmds []proxy_config.Cmd,
	maxConcurrent uint32) func(http.ResponseWriter, *http.Request) {
	return func(rw http.ResponseWriter, req *http.Request) {
		httpStatusCode, errCode, err := proxy_hook.CallBeforeRequest(req)
		if err != nil {
			writeReponse(rw, httpStatusCode, nil,
				proxy_message.CreateError(errCode, err.Error()))
			return
		}

		defer func() {
			if r := recover(); r != nil {
				log.Println("ufop error:", r)
				writeReponse(rw, http.StatusInternalServerError, nil,
					proxy_message.CreateError(101, "ufop server error"))
			}
		}()

		statusCode, data := process(req, client, resource, cmds)

		header, data := proxy_hook.CallAfterRequest(req, data)
		writeReponse(rw, statusCode, header, data)
	}
}

func createHealthHandler() func(http.ResponseWriter, *http.Request) {
	return func(rw http.ResponseWriter, req *http.Request) {
		rw.Write([]byte(`{"code":0, "message":"hi"}`))
	}
}

var (
	maxConcurrent uint32
)

func MaxConcurrent() uint32 {
	return maxConcurrent
}

func Run(port int,
	client proxy_client.Client,
	resource proxy_resource.Resource,
	cmds []proxy_config.Cmd,
	maxConcurrentP uint32) {
	maxConcurrent = maxConcurrentP
	http.HandleFunc("/handler", createHandler(client, resource, cmds, maxConcurrent))
	http.HandleFunc("/health", createHealthHandler())

	log.Fatalln(http.ListenAndServe("0.0.0.0:"+strconv.Itoa(port), nil))
}
