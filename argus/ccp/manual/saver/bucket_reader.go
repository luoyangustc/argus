package saver

import (
	"bufio"
	"compress/gzip"
	"context"
	"fmt"
	"io"
	"strings"

	xlog "github.com/qiniu/xlog.v1"
	BUCKET "qiniu.com/argus/argus/com/bucket"
	"qiniupkg.com/api.v7/kodo"
)

////////////////////////////////////////////////////////////////////////////////

type LineIter interface {
	Next(ctx context.Context) (string, bool, error)
	Error() error
	Close() error
}

func NewBucketKeyIter(cfg kodo.Config, ak, sk string, bucket, domain, key string, gzip bool) LineIter {
	return &KeyIter{
		Bucket: BUCKET.Bucket{
			Config: BUCKET.Config{
				Config: cfg,
				Bucket: bucket,
				Domain: domain,
			}.New(ak, sk, 0, bucket, ""),
		},
		Key:  key,
		Gzip: gzip,
	}
}

var _ LineIter = &KeyIter{}

type KeyIter struct {
	BUCKET.Bucket
	Key  string
	Gzip bool

	r       io.ReadCloser
	gr      io.ReadCloser
	scanner *bufio.Scanner

	end bool
	err error
}

func (iter *KeyIter) Next(ctx context.Context) (string, bool, error) {
	var (
		xl  = xlog.FromContextSafe(ctx)
		err error
	)

	if iter.end {
		xl.Info("iter.end")
		return "", false, nil
	}

	// init scanner
	if iter.scanner == nil {
		iter.r, iter.err = iter.Bucket.ReadByDomain(ctx, iter.Key)
		xl.Infof("iter.Bucket.Read: %s, %v", iter.Key, iter.err)

		if iter.err != nil {
			iter.end = true
			return "", false, nil
		}

		xl.Infof("iter.Gzip, %v", iter.Gzip)
		if iter.Gzip {
			iter.gr, err = gzip.NewReader(iter.r)
			if err != nil {
				xl.Errorf("gzip.NewReader error: %#v", err.Error())
				return "", false, err
			}
			iter.scanner = bufio.NewScanner(iter.gr)
		} else {
			iter.scanner = bufio.NewScanner(iter.r)
		}
	}

	// do scan finish
	if !iter.scanner.Scan() {
		return "", false, nil
	}

	return strings.TrimSpace(iter.scanner.Text()), true, nil
}

func (iter *KeyIter) Error() error { return iter.err }
func (iter *KeyIter) Close() (err error) {
	defer func() {
		if err0 := recover(); err0 != nil {
			err = fmt.Errorf("%v", err0)
		}
	}()

	if iter.Gzip && iter != nil && iter.gr != nil {
		iter.gr.Close()
		iter.gr = nil
	}

	if iter.r != nil {
		iter.r.Close()
		iter.r = nil
	}

	return err
}
