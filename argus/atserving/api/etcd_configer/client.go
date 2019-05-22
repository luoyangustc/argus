package client

import (
	"context"
	"encoding/json"
	"fmt"
	etcd "github.com/coreos/etcd/clientv3"
	"github.com/pkg/errors"
	"regexp"
	"time"
)

type Client struct {
	cfg  *Config
	etcd *etcd.Client
}

type Config struct {
	KeyPrefix string
	Etcd      etcd.Config
}

func New(cfg Config) (c *Client, err error) {
	c = &Client{
		cfg: &cfg,
	}
	c.etcd, err = etcd.New(cfg.Etcd)
	return
}

type PostReleaseArgs struct {
	App      string   `json:"app"`
	Desc     string   `json:"desc"`
	MetaData MetaData `json:"metadata"`
}

type PostReleaseResp struct {
	Version string `json:"version"`
}

func (c Client) PostRelease(ctx context.Context, args PostReleaseArgs) (resp *PostReleaseResp, err error) {
	app := args.App
	if !regexp.MustCompile(`^[a-z][a-z0-9_-]{3,19}$`).MatchString(app) {
		return nil, fmt.Errorf("invalid app name")
	}
	version := genVersion()
	release := Release{
		App:        app,
		Version:    version,
		MetaData:   args.MetaData,
		CreateTime: time.Now(),
		Desc:       args.Desc,
	}
	key := c.cfg.KeyPrefix + fmt.Sprintf("/%s/%s", app, version)
	val, err := json.Marshal(release)
	if err != nil {
		return nil, errors.Wrap(err, "json.Marshal")
	}
	_, err = c.etcd.Put(ctx, key, string(val))
	if err != nil {
		return nil, errors.Wrap(err, "etcd.Put")
	}
	return &PostReleaseResp{
		Version: version,
	}, nil
}

type GetReleaseResp struct {
	CreateTime time.Time `json:"create_time"`
	MetaData   MetaData  `json:"metadata"`
	Desc       string    `json:"desc"`
}

func (c Client) GetRelease(ctx context.Context, app, version string) (resp *GetReleaseResp, err error) {
	var release Release
	key := c.cfg.KeyPrefix + fmt.Sprintf("/%s/%s", app, version)
	val, err := c.etcd.Get(ctx, key)
	if err != nil {
		return nil, errors.Wrap(err, "etcd.Get")
	}
	if len(val.Kvs) != 1 {
		return nil, errors.Wrap(err, "bad etcd key")
	}
	err = json.Unmarshal(val.Kvs[0].Value, &release)
	if err != nil {
		return nil, errors.Wrap(err, "json.Unmarshal")
	}
	return &GetReleaseResp{
		CreateTime: release.CreateTime,
		MetaData:   release.MetaData,
		Desc:       release.Desc,
	}, nil
}

type ListReleaseResp struct {
	Version    string    `json:"version"`
	CreateTime time.Time `json:"create_time"`
	Desc       string    `json:"desc"`
}

func (c Client) ListRelease(ctx context.Context, app string) (resp []ListReleaseResp, err error) {
	r := make([]ListReleaseResp, 0)
	key := c.cfg.KeyPrefix + fmt.Sprintf("/%s", app)
	val, err := c.etcd.Get(ctx, key, etcd.WithPrefix())
	if err != nil {
		return nil, errors.Wrap(err, "etcd.Get")
	}
	for _, v := range val.Kvs {
		var release Release
		err = json.Unmarshal(v.Value, &release)
		if err != nil {
			return nil, errors.Wrap(err, "json.Unmarshal")
		}
		r = append(r, ListReleaseResp{
			Version:    release.Version,
			CreateTime: release.CreateTime,
			Desc:       release.Desc,
		})
	}
	return r, nil
}

func genVersion() string {
	return time.Now().Format("2006-01-02-15-04-05")
}
