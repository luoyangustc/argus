package util

import (
	"strings"

	minio "github.com/minio/minio-go"
)

type FileSaver interface {
	URI(url string) string
}

type MinioConfig struct {
	Host   string `json:"endpoint"`
	AK     string `json:"ak"`
	SK     string `json:"sk"`
	Bucket string `json:"bucket"`
	Prefix string `json:"prefix"`
}

type minioSaver struct {
	*minio.Client
	MinioConfig
	host string
}

func NewMinioSaver(config MinioConfig) (FileSaver, error) {
	// 处理host
	host := strings.ToLower(config.Host)
	host = strings.TrimPrefix(host, "http://")
	host = strings.TrimPrefix(host, "https://")

	// 新建minio client
	client, err := minio.New(host, config.AK, config.SK, false)
	if err != nil {
		return nil, err
	}

	// NOTE：目前没有写入操作，暂不确认minio可访问性
	// // 确认bucket是否存在
	// exists, err := client.BucketExists(config.Bucket)
	// if err != nil {
	// 	return nil, err
	// }
	// if !exists {
	// 	return nil, errors.New("bucket not exist")
	// }

	return &minioSaver{
		Client:      client,
		MinioConfig: config,
		host:        "http://" + host,
	}, nil
}

func (s *minioSaver) URI(url string) string {
	if IsHttpUrl(url) {
		return url
	}

	if !strings.HasPrefix(url, "/") {
		url = "/" + url
	}
	return s.host + url
}
