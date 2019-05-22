package feature_group

import (
	"bytes"
	"context"
	"crypto/sha1"
	"encoding/base64"
	"encoding/hex"
	"fmt"
	"io"
	"path"
	"strconv"
	"strings"

	"qiniupkg.com/api.v7/kodo"
	"qiniupkg.com/api.v7/kodocli"

	URI "qiniu.com/argus/argus/com/uri"
	STS "qiniu.com/argus/sts/client"
)

type SaverConfig struct {
	Kodo struct {
		Config *kodo.Config `json:"config"`
		Zone   int          `json:"zone"`
		UID    uint32       `json:"uid"`
		Bucket string       `json:"bucket"`
		Prefix string       `json:"prefix"`
	} `json:"kodo"`
}

type Saver interface {
	Save(ctx context.Context, uid uint32, gid, uri string) (string, error)
}

type KodoSaver struct {
	SaverConfig
	sts STS.Client
}

func NewKodoSaver(conf SaverConfig, sts STS.Client) *KodoSaver {
	return &KodoSaver{
		SaverConfig: conf,
		sts:         sts,
	}
}

func (s *KodoSaver) up(ctx context.Context, key string, r io.Reader, length int64) error {

	key = path.Join(s.SaverConfig.Kodo.Prefix, key)
	client := kodo.New(s.SaverConfig.Kodo.Zone, s.SaverConfig.Kodo.Config)
	token, err := client.MakeUptokenWithSafe(
		&kodo.PutPolicy{
			Scope:   s.SaverConfig.Kodo.Bucket + ":" + key,
			UpHosts: s.SaverConfig.Kodo.Config.UpHosts,
		})
	if err != nil {
		return err
	}
	uploader := kodocli.NewUploader(
		s.SaverConfig.Kodo.Zone,
		&kodocli.UploadConfig{UpHosts: s.SaverConfig.Kodo.Config.UpHosts},
	)
	ret := kodo.PutRet{}
	err = uploader.Put(ctx, &ret, token, key, r, length, nil)
	return err
}

func (s *KodoSaver) Save(ctx context.Context, uid uint32, gid, uri string) (string, error) {
	var (
		r      io.Reader
		length int64
	)
	switch {
	case strings.HasPrefix(uri, URI.DataURIPrefix):
		bs, err := base64.StdEncoding.DecodeString(strings.TrimPrefix(uri, URI.DataURIPrefix))
		if err != nil {
			return "", err
		}
		r = bytes.NewReader(bs)
		length = int64(len(bs))
	default:
		var err error
		r, length, _, err = s.sts.Get(ctx, uri, nil)
		if err != nil {
			return "", err
		}
	}

	h := sha1.New()
	h.Write([]byte(uri))
	key := path.Join(
		strconv.FormatUint(uint64(uid), 10),
		gid, hex.EncodeToString(h.Sum(nil)),
	)
	err := s.up(ctx, key, r, length)
	if err != nil {
		return "", err
	}

	return fmt.Sprintf("qiniu:///%s/%s",
		s.SaverConfig.Kodo.Bucket,
		path.Join(s.SaverConfig.Kodo.Prefix, key),
	), nil
}
