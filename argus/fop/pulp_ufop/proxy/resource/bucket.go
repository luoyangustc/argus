package proxy_resource

import (
	"errors"
	"log"

	"qiniu.com/argus/fop/pulp_ufop/proxy/config"

	"qiniupkg.com/api.v7/conf"
	"qiniupkg.com/api.v7/kodo"
	"qiniupkg.com/api.v7/kodocli"
)

type bucket struct {
	proxy_config.Bucket
}

func (bucket *bucket) Upload(filepath, key string) (string, error) {
	//初始化AK，SK
	conf.ACCESS_KEY = bucket.Ak
	conf.SECRET_KEY = bucket.Sk

	//创建一个Client
	c := kodo.New(0, nil)

	//设置上传的策略
	policy := &kodo.PutPolicy{
		Scope: bucket.Name + ":" + key,
		//设置Token过期时间
		Expires: 3600,
	}
	//生成一个上传token
	token := c.MakeUptoken(policy)

	//构建一个uploader
	zone := 0
	uploader := kodocli.NewUploader(zone, nil)

	var ret struct {
		Hash string `json:"hash"`
		Key  string `json:"key"`
	}
	//设置上传文件的路径
	//调用PutFile方式上传，这里的key需要和上传指定的key一致
	err := uploader.PutFile(nil, &ret, token, key, filepath, nil)
	//打印出错信息
	if err != nil {
		return "", err
	}
	return "http://" + bucket.Domain + "/" + key, nil
}

func (bucket *bucket) Delete(key string) error {
	log.Println("delete ", key)
	conf.ACCESS_KEY = bucket.Ak
	conf.SECRET_KEY = bucket.Sk
	c := kodo.New(0, nil)
	p := c.Bucket(bucket.Name)

	return p.Delete(nil, key)
}

func CreateBucketResource(b *proxy_config.Bucket) (Resource, error) {
	if b.Ak == "" {
		return nil, errors.New("no ak")
	}

	if b.Sk == "" {
		return nil, errors.New("no sk")
	}

	if b.Name == "" {
		return nil, errors.New("no name")
	}

	if b.Domain == "" {
		return nil, errors.New("no domain")
	}
	return &bucket{*b}, nil
}
