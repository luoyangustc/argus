package main

import (
	"crypto/hmac"
	"crypto/sha1"
	"encoding/base64"
	"fmt"
	"net/http"
	"net/http/httputil"
	"net/url"
	"strconv"
	"strings"
	"time"

	"qbox.us/cc/config"

	_httputil "github.com/qiniu/http/httputil.v1"
	"github.com/qiniu/http/restrpc.v1"
	"github.com/qiniu/http/servestk.v1"
	"github.com/qiniu/log.v1"
)

type ConfigServer struct {
	HTTPHost   string `json:"http_host"`
	DebugLevel int    `json:"debug_level"`
	Src        struct {
		User struct {
			AK string `json:"ak"`
			SK string `json:"sk"`
		} `json:"user"`
	} `json:"src"`
}

func serverMain() {

	var conf ConfigServer
	if err := config.Load(&conf); err != nil {
		log.Fatal("Failed to load configure file!")
	}

	log.SetOutputLevel(conf.DebugLevel)
	// log.Debugf("load conf %#v", conf)

	mux := servestk.New(restrpc.NewServeMux())
	mux.SetDefault(http.HandlerFunc(HandlerFile(conf.Src.User.AK, conf.Src.User.SK)))

	if err := http.ListenAndServe(conf.HTTPHost, mux); err != nil {
		log.Errorf("start error: %v", err)
	}

}

/*

GET /domain/key HTTP/1.1
Host: xxx

==>

GET /key HTTP/1.1
Host: domain

*/
func HandlerFile(accessKey, secretKey string) func(w http.ResponseWriter, r *http.Request) {

	return func(w http.ResponseWriter, r *http.Request) {

		strs := strings.SplitN(r.URL.Path, "/", 3)
		if len(strs) < 3 {
			_httputil.ReplyErr(w, http.StatusBadRequest, "bad domain and key")
			return
		}

		director := func(req *http.Request) {
			if _, ok := req.Header["User-Agent"]; !ok {
				// explicitly disable User-Agent so it's not set to default value
				req.Header.Set("User-Agent", "")
			}

			req.Host = strs[1]
			{
				query := req.URL.Query()
				query.Del("token")
				query.Set("e", strconv.FormatInt(time.Now().Add(time.Hour).Unix(), 10))
				_url := fmt.Sprintf("http://%s/%s?%s", strs[1], strs[2], query.Encode())
				h := hmac.New(sha1.New, []byte(secretKey))
				h.Write([]byte(_url))
				sign := base64.URLEncoding.EncodeToString(h.Sum(nil))
				_url += "&token=" + fmt.Sprintf("%s:%s", accessKey, sign)
				req.URL, _ = url.Parse(_url)
			}

		}
		proxy := &httputil.ReverseProxy{Director: director}
		proxy.ServeHTTP(w, r)

	}

}
