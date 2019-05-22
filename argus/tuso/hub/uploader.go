package hub

import (
	"bytes"
	"context"

	"qiniupkg.com/api.v7/kodo"
	"qiniupkg.com/api.v7/kodocli"
)

type uploader interface {
	upload(ctx context.Context, key string, buf []byte) error
}
type kodoUploader struct {
	cfg KodoConfig
}

type KodoConfig struct {
	Ak     string   `json:"ak"`
	Sk     string   `json:"sk"`
	Bucket string   `json:"bucket"`
	Region int      `json:"region"`
	Domain string   `json:"domain"`
	Prefix string   `json:"prefix"`
	IoHost []string `json:"io_host"`
	UpHost []string `json:"up_host"`
}

func (k *kodoUploader) upload(ctx context.Context, key string, buf []byte) error {

	key2 := k.cfg.Prefix + "/" + key
	xl.Info("upload", key2)

	c := kodo.New(0, &kodo.Config{
		AccessKey: k.cfg.Ak,
		SecretKey: k.cfg.Sk,
	})
	// https://developer.qiniu.com/kodo/manual/1206/put-policy
	// <bucket>:<key>，表示只允许用户上传指定 key 的文件。在这种格式下文件默认允许修改，若已存在同名资源则会被覆盖。如果只希望上传指定 key 的文件，并且不允许修改，那么可以将下面的 insertOnly 属性值设为 1。
	// 本地开发环境和CS环境经常清空数据库，需要允许文件覆盖
	token, err := c.MakeUptokenWithSafe(&kodo.PutPolicy{
		Scope:   k.cfg.Bucket + ":" + key2,
		UpHosts: k.cfg.UpHost,
	})

	if err != nil {
		return err
	}
	uploader := kodocli.NewUploader(-1, &kodocli.UploadConfig{UpHosts: k.cfg.UpHost})
	ret := kodo.PutRet{}

	err = uploader.Put(ctx, &ret, token, key2, bytes.NewReader(buf), int64(len(buf)), nil)
	return err
}
