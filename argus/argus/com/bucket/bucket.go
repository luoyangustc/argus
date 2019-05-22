package bucket

import (
	"context"
	"io"
	"net/http"
	"net/url"
	"os"
	"path"

	"github.com/pkg/errors"
	"github.com/qiniu/rpc.v3"
	xlog "github.com/qiniu/xlog.v1"
	"qbox.us/api/v2/rs"
	"qiniu.com/auth/qboxmac.v1"
	"qiniupkg.com/api.v7/kodo"
	"qiniupkg.com/api.v7/kodocli"
)

type Storage interface {
	List(ctx context.Context) ([]string, error)                                      // return keys
	Save(ctx context.Context, name string, buf io.Reader, len int64) (string, error) // return key
	Read(ctx context.Context, key string) (io.ReadCloser, error)
	ReadByDomain(ctx context.Context, key string) (io.ReadCloser, error)
}

var _ Storage = Bucket{}

type Bucket struct {
	Config
}

func (b Bucket) List(ctx context.Context) ([]string, error) {

	var (
		marker string
		limit  = 50
		err    error
	)

	xl := xlog.FromContextSafe(ctx)
	cli := kodo.New(b.Zone, &b.Config.Config)
	bCli, _ := cli.BucketWithSafe(b.Bucket)

	xl.Infof("BucketList, prefix = %s", b.Prefix)

	var keys = make([]string, 0)
	for {
		var (
			entries []kodo.ListItem
		)
		entries, _, marker, err = bCli.List(ctx, b.Prefix, "", marker, limit)
		for _, entry := range entries {
			keys = append(keys, entry.Key)
		}
		if err != nil {
			if err == io.EOF {
				err = nil
			}
			break
		}
	}

	return keys, err
}

func (b Bucket) Save(ctx context.Context, name string, r io.Reader, length int64) (string, error) {

	xl := xlog.FromContextSafe(ctx)
	key := b.Config.Prefix + name
	client := kodo.New(b.Config.Zone, &b.Config.Config)
	token, err := client.MakeUptokenWithSafe(
		&kodo.PutPolicy{
			Scope:   b.Config.Bucket + ":" + key,
			UpHosts: b.Config.Config.UpHosts,
		})
	if err != nil {
		return "", err
	}
	uploader := kodocli.NewUploader(
		b.Config.Zone,
		&kodocli.UploadConfig{UpHosts: b.Config.Config.UpHosts},
	)
	ret := kodo.PutRet{}
	err = uploader.Put(ctx, &ret, token, key, r, length, nil)
	xl.Infof("BucketSave, %s, %d, ret = %+v, %+v", key, length, ret, err)
	if err != nil {
		return "", err
	}
	return key, err
}

func (b Bucket) Read(ctx context.Context, key string) (io.ReadCloser, error) {

	rsClient := rs.New(qboxmac.NewTransport(
		&qboxmac.Mac{
			AccessKey: b.AccessKey,
			SecretKey: []byte(b.SecretKey),
		},
		nil,
	))

	encodeKey := b.Bucket + ":" + key
	ret, _, err := rsClient.Get(encodeKey, "")
	if err != nil {
		return nil, errors.Wrap(err, "rs1.Get")
	}

	req, err := http.NewRequest("GET", ret.URL, nil)
	if err != nil {
		return nil, errors.Wrap(err, "getHTTP http.NewRequest")
	}
	req = req.WithContext(ctx)
	_resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return nil, errors.Wrap(err, "getHTTP client.Do")
	}
	if _resp.StatusCode/100 != 2 {
		defer _resp.Body.Close()
		return nil, rpc.ResponseError(_resp)
	}
	return _resp.Body, err
}

func (b Bucket) ReadByDomain(ctx context.Context, key string) (io.ReadCloser, error) {

	xl := xlog.FromContextSafe(ctx)
	xl.Infof("domain = %s, key = %s", b.Domain, key)
	xl.Infof("IoHost, %v", b.Config.Config.IoHost)
	var uri = kodo.New(0, &b.Config.Config).MakePrivateUrl(kodo.MakeBaseUrl(b.Domain, key), nil)
	if b.Config.Config.IoHost != "" {
		uri2, _ := url.Parse(uri)
		uri2.Host = b.Config.Config.IoHost
		uri = uri2.String()
	}

	xl.Infof("url, %s", uri)

	req, err := http.NewRequest("GET", uri, nil)
	if err != nil {
		return nil, errors.Wrap(err, "getHTTP http.NewRequest")
	}
	if b.Config.Config.IoHost != "" {
		req.Host = b.Domain
	}
	req = req.WithContext(ctx)
	_resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return nil, errors.Wrap(err, "getHTTP client.Do")
	}
	if _resp.StatusCode/100 != 2 {
		defer _resp.Body.Close()
		return nil, rpc.ResponseError(_resp)
	}
	return _resp.Body, err
}

//----------------------------------------------------------------------------//

var _ Storage = LocalFS{}

type LocalFS struct {
	dir string
}

func NewLocalFS(dir string) LocalFS { return LocalFS{dir: dir} }

func (fs LocalFS) List(ctx context.Context) ([]string, error) {
	dir, _ := os.Open(fs.dir)
	return dir.Readdirnames(0)
}

func (fs LocalFS) Save(ctx context.Context, name string, r io.Reader, length int64) (string, error) {
	fname := path.Join(fs.dir, name)
	f, _ := os.OpenFile(fname, os.O_CREATE|os.O_RDWR, 0755)
	defer f.Close()
	_, _ = io.Copy(f, r)
	return fname, nil
}

func (fs LocalFS) Read(ctx context.Context, key string) (io.ReadCloser, error) {
	return os.Open(path.Join(fs.dir, key))
}

func (fs LocalFS) ReadByDomain(ctx context.Context, key string) (io.ReadCloser, error) {
	return os.Open(path.Join(fs.dir, key))
}
