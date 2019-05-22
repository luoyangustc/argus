package file_scan

import (
	"bufio"
	"context"
	"fmt"
	"io"
	"strings"
	"time"

	"qiniupkg.com/api.v7/kodo"

	"qiniu.com/argus/argus/com/bucket"
	ahttp "qiniu.com/argus/argus/com/http"
)

type IScanner interface {
	Scan(context.Context) (IIter, error)
	Count(context.Context) (int, error)
}

type KeyItem struct {
	URI string `json:"uri"`
}

type IIter interface {
	Next(context.Context) (KeyItem, bool)
	Error() error
}

////////////////////////////////////////////////////////////////////////////////

type Config struct {
	kodo.Config
	Bucket string   `json:"bucket"`
	Keys   []string `json:"keys"`
}

func (conf Config) New(ak, sk string, bucket string, keys []string) Config {
	return Config{
		Config: kodo.Config{
			AccessKey: ak,
			SecretKey: sk,
			RSHost:    conf.RSHost,
			RSFHost:   conf.RSFHost,
			APIHost:   conf.APIHost,
			Scheme:    conf.Scheme,
			IoHost:    conf.IoHost,
			UpHosts:   conf.UpHosts,
			Transport: conf.Transport,
		},
		Bucket: bucket,
		Keys:   keys,
	}
}

var _ IScanner = Scanner{}

type Scanner struct {
	Config
}

var _ IIter = &EntryIter{}

type EntryIter struct {
	RS []func() (io.ReadCloser, error)

	i, j    int
	r       io.ReadCloser
	scanner *bufio.Scanner

	count int
	end   bool
	err   error
}

func (iter *EntryIter) Next(ctx context.Context) (KeyItem, bool) {
	if iter.end {
		return KeyItem{}, false
	}

	if iter.scanner == nil {
		if iter.i >= len(iter.RS) {
			return KeyItem{}, false
		}

		var err error
		iter.r, err = iter.RS[iter.i]()
		if err != nil {
			iter.end = true
			iter.err = err
			return KeyItem{}, false
		}
		iter.scanner = bufio.NewScanner(iter.r)
	}

	if !iter.scanner.Scan() {
		iter.r.Close()
		iter.r = nil
		iter.j = 0
		iter.i++

		if err := iter.scanner.Err(); err != nil {
			iter.end = true
			iter.err = err
			iter.scanner = nil
			return KeyItem{}, false
		}
		iter.scanner = nil
		return iter.Next(ctx)
	}

	line := strings.TrimSpace(iter.scanner.Text())
	if len(line) == 0 {
		iter.j++
		return iter.Next(ctx)
	}

	iter.count++
	iter.j++
	return KeyItem{URI: line}, true
}

func (iter *EntryIter) Error() error { return iter.err }

func (b Scanner) Scan(ctx context.Context) (IIter, error) {
	rs := make([]func() (io.ReadCloser, error), 0, len(b.Keys))
	for _, key0 := range b.Keys {
		var key = key0
		rs = append(rs, func() (io.ReadCloser, error) {

			cli := ahttp.NewQiniuAuthRPCClient(b.AccessKey, b.SecretKey, time.Second*10)
			var domains = []struct {
				Domain string `json:"domain"`
				Tbl    string `json:"tbl"`
				Global bool   `json:"global"`
			}{}
			err := cli.Call(context.Background(), &domains,
				"GET", fmt.Sprintf("%s/v7/domain/list?tbl=%s", b.APIHost, b.Bucket),
			)
			if err != nil {
				return nil, err
			}

			return bucket.Bucket{
				Config: bucket.Config{
					Config: b.Config.Config, Bucket: b.Bucket, Domain: domains[0].Domain,
				},
			}.ReadByDomain(ctx, key)
		})
	}
	return &EntryIter{RS: rs}, nil
}

func (b Scanner) Count(ctx context.Context) (int, error) {
	iter, _ := b.Scan(ctx)
	for {
		_, ok := iter.Next(ctx)
		if !ok {
			break
		}
	}
	return iter.(*EntryIter).count, iter.Error()
}
