package main

import (
	"context"
	"encoding/json"
	"fmt"
	"time"

	xlog "github.com/qiniu/xlog.v1"
	"qbox.us/qconf/qconfapi"
	"qiniu.com/argus/argus/com/auth"
	ahttp "qiniu.com/argus/argus/com/http"
)

func Aksk(qconf *qconfapi.Config, uid uint32) (string, string, error) {
	ak, sk, err := auth.AkSk(qconfapi.New(qconf), uid)
	if err != nil {
		return "", "", err
	}
	return ak, sk, err
}

func InnerPostByAkSk(ctx context.Context, qconf *qconfapi.Config, uid uint32, url string,
	req interface{}, ret interface{}) error {
	xl := xlog.FromContextSafe(ctx)

	ak, sk, err := auth.AkSk(qconfapi.New(qconf), uid)
	if err != nil {
		xl.Errorf("AkSk err, %+v", err)
		return err
	}

	client := ahttp.NewQboxAuthRPCClient(ak, sk, time.Second*60)
	err = client.CallWithJson(ctx, ret, "POST", url, req)

	if err != nil {
		xl.Infof("InnerPostByAkSk, %+v, %+v, %s, %+v", req, ret, url, err)
	}
	return err
}

func InnerPost(ctx context.Context, uid uint32, url string,
	req interface{}, ret interface{}) error {
	xl := xlog.FromContextSafe(ctx)

	client := ahttp.NewQiniuStubRPCClient(uid, 1, time.Second*5)
	err := client.CallWithJson(ctx, ret, "POST", url, req)

	xl.Infof("InnerPost, %+v, %+v, %s, %+v", req, ret, url, err)
	return err
}

func InnerGet(ctx context.Context, uid uint32, url string,
	req interface{}, ret interface{}) error {
	xl := xlog.FromContextSafe(ctx)

	client := ahttp.NewQiniuStubRPCClient(uid, 1, time.Second*5)
	err := client.CallWithJson(ctx, ret, "GET", url, req)

	xl.Infof("InnerGet, %+v, %+v, %s, %+v", req, ret, url, err)
	return err
}

// ConvByJson
func ConvByJson(src interface{}, dest interface{}) error {
	tmpbs, err := json.Marshal(src)
	if err != nil {
		return err
	}
	return json.Unmarshal(tmpbs, dest)
}

// JsonStr
func JsonStr(obj interface{}) string {
	raw, err := json.Marshal(obj)
	if err != nil {
		return ""
	}
	return string(raw)
}

// Countdown 倒计时/秒
func Countdown(secs int, handler func(secs int)) {

	if secs > 9999 {
		secs = 9999
	}

	if handler == nil {
		handler = func(secs int) {
			fmt.Printf("countdown:%4d\r", secs)
		}
	}

	for secs > 0 {
		handler(secs)
		time.Sleep(time.Second)
		secs--
	}
}

// CountdownWithTips 倒计时/秒
func CountdownWithTips(secs int, tips string) {

	if secs > 9999 {
		secs = 9999
	}

	handler := func(secs int) {
		fmt.Printf("%s:%4d\r", tips, secs)
	}

	for secs > 0 {
		handler(secs)
		time.Sleep(time.Second)
		secs--
	}
}
