package video

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"io/ioutil"
	"net/http"
	"os"
	"path"
	"strings"
	"sync"
	"time"

	"github.com/qiniu/rpc.v3"
	xlog "github.com/qiniu/xlog.v1"
	"qbox.us/net/httputil"
	"qbox.us/qconf/qconfapi"
	"qiniu.com/argus/argus/com/auth"
	URI "qiniu.com/argus/argus/com/uri"
	"qiniupkg.com/api.v7/kodo"
	"qiniupkg.com/api.v7/kodocli"
)

type Saver interface {
	Save(ctx context.Context, offset int64, uri string) (string, error)
}

type SaverOPHook interface {
	Get(ctx context.Context, op string) (Saver, error)
}

type SaverHook interface {
	Get(ctx context.Context, uid uint32, vid string, params json.RawMessage) (SaverOPHook, error)
}

type SaverOPFunc func(ctx context.Context, op string) (Saver, error)

func (f SaverOPFunc) Get(ctx context.Context, op string) (Saver, error) {
	return f(ctx, op)
}

type SaverFunc func(ctx context.Context, offset int64, uri string) (string, error)

func (f SaverFunc) Save(ctx context.Context, offset int64, uri string) (string, error) {
	return f(ctx, offset, uri)
}

///////////////////////////////////////////////////////////////////////////////
type KodoSaveConfig struct {
	// save to kodo
	Qconf qconfapi.Config `json:"qconf"`
	Kodo  kodo.Config     `json:"kodo"` // UpHosts
}

var _ SaverHook = (*kodoSaver)(nil)

type kodoSaver struct {
	KodoSaveConfig
	*qconfapi.Client
}

func NewKodoSaver(config KodoSaveConfig) *kodoSaver {
	return &kodoSaver{
		KodoSaveConfig: config,
		Client:         qconfapi.New(&config.Qconf),
	}
}

func (s *kodoSaver) Get(ctx context.Context, uid uint32, vid string, params json.RawMessage) (SaverOPHook, error) {
	var (
		saveParams struct {
			Uid    uint32 `json:"uid"`
			Bucket string `json:"bucket"`
			Zone   int    `json:"zone"`
			Prefix string `json:"prefix"`
		}
		kodoCfg kodo.Config
		xl      = xlog.FromContextSafe(ctx)
	)
	err := json.Unmarshal(params, &saveParams)
	if err != nil {
		return nil, fmt.Errorf("parse save params failed: %v", err)
	}

	{
		if saveParams.Uid > 0 {
			uid = saveParams.Uid
		}
		ak, sk, err := auth.AkSk(s.Client, uid)
		if err != nil {
			return nil, fmt.Errorf("get aksk failed: %d %v", uid, err)
		}
		kodoCfg = kodo.Config{
			AccessKey: ak,
			SecretKey: sk,
		}
	}
	saveKeys := make(map[string]bool) // 保存过或者保存中的key的列表
	var lock sync.Mutex
	hook := SaverOPFunc(func(ctx context.Context, op string) (Saver, error) {
		saver := SaverFunc(func(ctx context.Context, offset int64, uri string) (string, error) {
			rc, length, err := readCloserFromURI(ctx, uri)
			if err != nil {
				return "", err
			}
			defer rc.Close()

			key := fmt.Sprintf("%s/%v", vid, offset)

			lock.Lock()
			isKeyExists := saveKeys[key]
			lock.Unlock()
			if isKeyExists {
				xl.Infof("saved key %q, skip", key)
			} else {
				lock.Lock()
				saveKeys[key] = true
				lock.Unlock()
				if err := s.up(ctx, kodoCfg, saveParams, key, rc, length); err != nil {
					return "", err
				}
			}

			if saveParams.Uid > 0 {
				return fmt.Sprintf("qiniu://%d@/%s/%s",
					saveParams.Uid, saveParams.Bucket,
					path.Join(saveParams.Prefix, key),
				), nil
			}
			return fmt.Sprintf("qiniu:///%s/%s",
				saveParams.Bucket,
				path.Join(saveParams.Prefix, key),
			), nil
		})
		return saver, nil
	})
	return hook, nil
}

func (s *kodoSaver) up(
	ctx context.Context,
	kodoCfg kodo.Config,
	saveParams struct {
		Uid    uint32 `json:"uid"`
		Bucket string `json:"bucket"`
		Zone   int    `json:"zone"`
		Prefix string `json:"prefix"`
	}, key string, r io.Reader, length int64) error {
	xl := xlog.FromContextSafe(ctx)
	client := kodo.New(saveParams.Zone, &kodoCfg)
	key = path.Join(saveParams.Prefix, key)
	xl.Infof("kodoSaver.up %v %#v %v", key, saveParams, length)
	token, err := client.MakeUptokenWithSafe(
		&kodo.PutPolicy{
			Scope:   saveParams.Bucket + ":" + key,
			UpHosts: s.KodoSaveConfig.Kodo.UpHosts,
		})
	if err != nil {
		return err
	}
	uploader := kodocli.NewUploader(
		saveParams.Zone,
		&kodocli.UploadConfig{UpHosts: s.KodoSaveConfig.Kodo.UpHosts},
	)
	ret := kodo.PutRet{}
	err = uploader.Put(ctx, &ret, token, key, r, length, nil)
	return err
}

///////////////////////////////////////////////////////////////////////////////
type FileSaveConfig struct {
	SaveSpace   string `json:"savespace"`
	SaveAddress string `json:"save_address"`
	DailyFolder bool   `json:"daily_folder"`
}

var _ SaverHook = (*fileSaver)(nil)

type fileSaver struct {
	FileSaveConfig
}

func NewFileSaver(config FileSaveConfig) *fileSaver {
	return &fileSaver{
		FileSaveConfig: config,
	}
}

func (s *fileSaver) Get(ctx context.Context, uid uint32, vid string, params json.RawMessage) (SaverOPHook, error) {
	var (
		saveParams struct {
			Prefix string `json:"prefix"`
		}
		xl = xlog.FromContextSafe(ctx)
	)
	err := json.Unmarshal(params, &saveParams)
	if err != nil {
		return nil, fmt.Errorf("parse save params failed: %v", err)
	}

	saveKeys := make(map[string]bool) // 保存过或者保存中的key的列表
	var lock sync.Mutex

	hook := SaverOPFunc(func(ctx context.Context, op string) (Saver, error) {
		saver := SaverFunc(func(ctx context.Context, offset int64, uri string) (string, error) {
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

			lock.Lock()
			isKeyExists := saveKeys[key]
			lock.Unlock()
			if isKeyExists {
				xl.Infof("saved key %q, skip", key)
				return s.SaveAddress + "/" + key, nil
			}
			lock.Lock()
			saveKeys[key] = true
			lock.Unlock()

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
		})
		return saver, nil
	})
	return hook, nil
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
