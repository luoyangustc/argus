package biz

import (
	"fmt"
	"os"

	c "qiniu.com/argus/test/configs"
	"qiniu.com/argus/test/lib/auth"
	"qiniu.com/argus/test/lib/qnhttp"
	"qiniu.com/argus/test/lib/util"
)

func DeleteBucket(user auth.AccessInfo, bucket string) *qnhttp.Response {
	url := c.Configs.Host.RS + "/drop/" + bucket
	token := auth.SignQboxToken(user, url, "")
	resp, err := qnhttp.New().Set("Authorization", "QBox "+token).Post(url, "", nil, nil)
	if resp.Status() == 200 {
		fmt.Println("删除bucket成功")
	} else {
		fmt.Println("删除失败" + err.Error())
	}
	return resp
}

func DeleteFile(user auth.AccessInfo, bucket, key string) (resp *qnhttp.Response, err error) {
	entry, _ := util.Encode(util.EncodeType_Base64URL, bucket+":"+key)
	url := c.Configs.Host.RS + "/delete/" + entry
	token := auth.SignQboxToken(user, url, "")
	s := qnhttp.New().Set("Authorization", "QBox "+token)
	resp, err = s.Post(url, nil, nil, nil)
	return
}

func V2BucketInfo(user auth.AccessInfo, bucket string) (*qnhttp.Response, error) {
	url := c.Configs.Host.UC + "/v2/bucketInfo?bucket=" + bucket
	token := auth.SignQboxToken(user, url, "")
	s := qnhttp.New().Set("Authorization", "QBox "+token)
	return s.Post(url, "", nil, nil)
}

type MkBucketArgs struct {
	User   auth.AccessInfo
	Bucket string
	Region string
	Global bool
	Line   bool
}

func MkBucket(args MkBucketArgs) (*qnhttp.Response, error) {
	url := c.Configs.Host.RS + "/mkbucket/" + args.Bucket
	url += "/region/"

	if args.Region != "" {
		url += args.Region
	} else {
		url += os.Getenv("TEST_ZONE")
	}

	token := auth.SignQboxToken(args.User, url, "")
	return qnhttp.New().Set("Authorization", "QBox "+token).Post(url, "", nil, nil)
}

type PrivateArgs struct {
	User    auth.AccessInfo
	Bucket  string
	Private string
}

func Private(args PrivateArgs) (*qnhttp.Response, error) {
	url := c.Configs.Host.UC + "/private?bucket=" + args.Bucket + "&private=" + args.Private
	token := auth.SignQboxToken(args.User, url, "")
	req := qnhttp.New().Set("Authorization", "QBox "+token)
	return req.Post(url, nil, nil, nil)
}

func Stat(user auth.AccessInfo, entry string) (*qnhttp.Response, error) {
	url := c.Configs.Host.RS + "/stat/" + util.EncodeURL(entry)

	token := auth.SignQboxToken(user, url, "")
	s := qnhttp.New().Set("Authorization", "QBox "+token)
	return s.Post(url, "", nil, nil)
}

func IsExist(user auth.AccessInfo, entry string) bool {
	resp, _ := Stat(user, entry)
	return resp.Status() == 200
}
