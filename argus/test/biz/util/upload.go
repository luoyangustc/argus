package util

import (
	"encoding/base64"
	"fmt"

	"qiniu.com/argus/test/configs"
	"qiniu.com/argus/test/lib/auth"
	"qiniu.com/argus/test/lib/qnhttp"
)

func FormUp(key, filePath, token string) (*qnhttp.Response, error) {
	body := map[string]string{"key": key, "filepath": filePath, "token": token}
	s := qnhttp.New().Set("Content-Type", "multipart/form-data")
	resp, err := s.Post(configs.Configs.Host.UP, body, nil, nil)
	return resp, err
}

//用于普通上传
func DoFormUp(user auth.AccessInfo, bucket, upKey, filePath string) (*qnhttp.Response, error) {
	//签token
	putPolicy := &auth.PutPolicy{
		Scope: bucket + ":" + upKey,
	}
	upToken := putPolicy.MakeUptoken(user.Key, user.Secret)
	resp, err := FormUp(upKey, filePath, upToken)
	if err != nil {
		fmt.Println(err.Error())
		return nil, err
	}
	return resp, err
}

//ccplink文件上传
func CcpUploadFile(upKey, filepath string) (*qnhttp.Response, error) {
	var user auth.AccessInfo
	var bucket string
	user.Key = configs.Configs.ArgusBcpTestbucket.User.AK
	user.Secret = configs.Configs.ArgusBcpTestbucket.User.SK
	bucket = configs.Configs.ArgusBcpTestbucket.Name
	resp, err := DoFormUp(user, bucket, upKey, filepath)
	if err != nil {
		fmt.Println(err.Error())
		return nil, err
	}
	return resp, err
}

//用于普通删除
func FileDel(bucket, key string) (*qnhttp.Response, error) {
	entry := bucket + ":" + key
	encodedEntryURI := base64.StdEncoding.EncodeToString([]byte(entry))
	uri := configs.Configs.Host.RS + "/delete/" + encodedEntryURI
	s := qnhttp.New().Set("Content-Type", "application/x-www-form-urlencoded")
	s.Header.Set("Authorization", fmt.Sprintf("QiniuStub uid=%d&ut=%d", 1, 0))
	return s.Post(uri, nil, nil, nil)
}
