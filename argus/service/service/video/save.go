package video

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"io/ioutil"
	"net/http"
	"os"
	"path"
	"strings"
	"time"

	"github.com/minio/minio-go"
	httputil "github.com/qiniu/http/httputil.v1"
	rpc "github.com/qiniu/rpc.v3"
	xlog "github.com/qiniu/xlog.v1"

	URI "qiniu.com/argus/argus/com/uri"
)

type Saver interface {
	Save(ctx context.Context, offsetMS int64, uri string) (string, error)
}

type SaverHook interface {
	Get(ctx context.Context, vid string, params json.RawMessage) (Saver, error)
}

type SaverFunc func(ctx context.Context, offsetMS int64, uri string) (string, error)

func (f SaverFunc) Save(ctx context.Context, offsetMS int64, uri string) (string, error) {
	return f(ctx, offsetMS, uri)
}

///////////////////////////////////////////////////////////////////////////////

type FileSaveConfig struct {
	SaveSpace   string `json:"save_space"`
	SaveAddress string `json:"save_address"`
	DailyFolder bool   `json:"daily_folder"`
}

var _ SaverHook = (*fileSaver)(nil)

type fileSaver struct {
	FileSaveConfig
}

func NewFileSaver(config FileSaveConfig) *fileSaver { return &fileSaver{FileSaveConfig: config} }

func (s *fileSaver) Get(
	ctx context.Context, vid string, params json.RawMessage,
) (Saver, error) {
	var (
		saveParams struct {
			Prefix string `json:"prefix"`
		}
	)
	err := json.Unmarshal(params, &saveParams)
	if err != nil {
		return nil, fmt.Errorf("parse save params failed: %v", err)
	}

	return SaverFunc(func(ctx context.Context, offset int64, uri string) (string, error) {
		var (
			key, fpath string
			rc         io.ReadCloser
			f          *os.File
			err        error
		)

		if len(saveParams.Prefix) > 0 {
			key = fmt.Sprintf("%s/%s/%v", saveParams.Prefix, vid, offset)
		} else {
			key = fmt.Sprintf("%s/%v", vid, offset)
		}

		if s.DailyFolder {
			key = path.Join(time.Now().Format("20060102"), key)
		}

		{
			rc, _, err = readCloserFromURI(ctx, uri)
			if err != nil {
				return "", err
			}
			defer rc.Close()
		}

		{
			fpath = path.Join(s.SaveSpace, key)
			dir := path.Dir(fpath)
			err = os.MkdirAll(dir, 0755)
			if err != nil {
				return "", err
			}

			f, err = os.Create(fpath)
			if err != nil {
				return "", err
			}
			defer f.Close()
		}

		if _, err = io.Copy(f, rc); err != nil {
			return "", err
		}
		return s.SaveAddress + "/" + key, nil
	}), nil
}

///////////////////////////////////////////////////////////////////////////////
func readCloserFromURI(ctx context.Context, uri string) (io.ReadCloser, int64, error) {
	var (
		rc     io.ReadCloser
		length int64
	)

	switch {
	case strings.HasPrefix(uri, URI.DataURIPrefix):
		bs, err := base64.StdEncoding.DecodeString(strings.TrimPrefix(uri, URI.DataURIPrefix))
		if err != nil {
			return nil, 0, err
		}
		rc = ioutil.NopCloser(bytes.NewReader(bs))
		length = int64(len(bs))
	case strings.HasPrefix(uri, "sts://"):
		uri = strings.Replace(uri, "sts", "http", 1)
		resp, err := rpc.DefaultClient.DoRequest(ctx, "GET", uri)
		if err != nil {
			return nil, 0, err
		}
		if resp.StatusCode != http.StatusOK {
			err = rpc.ResponseError(resp)
			code := resp.StatusCode
			resp.Body.Close()
			return nil, 0, httputil.NewError(code, err.Error())
		}
		rc, length = resp.Body, resp.ContentLength
	default:
		return nil, 0, fmt.Errorf("unsupported uri format: %s", uri)
	}

	return rc, length, nil
}

///////////////////////////////////////////////////////////////////////////////

type MinioConfig struct {
	Host   string `json:"endpoint"`
	AK     string `json:"ak"`
	SK     string `json:"sk"`
	Bucket string `json:"bucket"`
	Prefix string `json:"prefix"`
}

var _ SaverHook = (*minioSaver)(nil)

type minioSaver struct {
	*minio.Client
	MinioConfig
}

func NewMinioSaver(config MinioConfig) (SaverHook, error) {
	// 处理host
	host := strings.ToLower(config.Host)
	host = strings.TrimPrefix(host, "http://")
	host = strings.TrimPrefix(host, "https://")

	// 新建minio client
	client, err := minio.New(host, config.AK, config.SK, false)
	if err != nil {
		return nil, err
	}

	// 确认bucket是否存在
	exists, err := client.BucketExists(config.Bucket)
	if err != nil {
		return nil, err
	}
	if !exists {
		return nil, errors.New("bucket not exist")
	}

	return &minioSaver{
		Client:      client,
		MinioConfig: config,
	}, nil
}

func (s *minioSaver) Get(
	ctx context.Context, vid string, params json.RawMessage,
) (Saver, error) {
	return SaverFunc(func(ctx context.Context, offset int64, uri string) (string, error) {
		var (
			xl   = xlog.FromContextSafe(ctx)
			path = fmt.Sprintf("%s/%s/%v.jpg", s.Prefix, vid, offset)
			rc   io.ReadCloser
			len  int64
			err  error
		)

		rc, len, err = readCloserFromURI(ctx, uri)
		if err != nil {
			return "", err
		}
		defer rc.Close()

		_, err = s.PutObject(s.Bucket, path, rc, len, minio.PutObjectOptions{})
		if err != nil {
			xl.Errorf("fail to upload cut(%d), %v", offset, err)
			return "", err
		}

		return fmt.Sprintf("/%s/%s", s.Bucket, path), nil
	}), nil
}
