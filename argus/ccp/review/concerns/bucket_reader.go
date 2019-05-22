package concerns

import (
	"bufio"
	"compress/gzip"
	"context"
	"io"
	"strings"

	xlog "github.com/qiniu/xlog.v1"
	BUCKET "qiniu.com/argus/argus/com/bucket"
	"qiniupkg.com/api.v7/kodo"
)

////////////////////////////////////////////////////////////////////////////////

type LineIter interface {
	Next(ctx context.Context) (string, bool)
	Error() error
	Close() error
}

var _ LineIter = &KeyIter{}

type KeyIter struct {
	BUCKET.Bucket
	Key  string
	Gzip bool

	r       io.ReadCloser
	gr      io.ReadCloser
	scanner *bufio.Reader

	end bool
	err error
}

func (iter *KeyIter) Next(ctx context.Context) (string, bool) {
	xl := xlog.FromContextSafe(ctx)

	if iter.end {
		xl.Info("iter.end")
		return "", false
	}

	// init scanner
	if iter.scanner == nil {
		iter.r, iter.err = iter.Bucket.ReadByDomain(ctx, iter.Key)
		xl.Infof("iter.Bucket.Read, %s, %v", iter.Key, iter.err)

		if iter.err != nil {
			iter.end = true
			return "", false
		}

		xl.Infof("iter.Gzip, %v", iter.Gzip)
		if iter.Gzip {
			iter.gr, _ = gzip.NewReader(iter.r)
			iter.scanner = bufio.NewReader(iter.gr)
		} else {
			iter.scanner = bufio.NewReader(iter.r)
		}
	}

	// do scan finish
	return iter.ReadLine()
}

func (iter *KeyIter) ReadLine() (string, bool) {
	var ret []byte

	for {
		line, isPrefix, err := iter.scanner.ReadLine()
		// scan finish
		if err != nil {
			return "", false
		}

		ret = append(ret, line...)

		if !isPrefix {
			break
		}
	}

	return strings.TrimSpace(string(ret)), true
}

func (iter *KeyIter) Error() error { return iter.err }
func (iter *KeyIter) Close() error {
	if iter.Gzip && iter.gr != nil {
		iter.gr.Close()
		iter.gr = nil
	}

	if iter.r != nil {
		iter.r.Close()
		iter.r = nil
	}

	return nil
}

func NewBucketKeyIter(cfg kodo.Config, ak, sk string, bucket, domain, key string) LineIter {
	return &KeyIter{
		Bucket: BUCKET.Bucket{
			Config: BUCKET.Config{
				Config: cfg,
				Bucket: bucket,
				Domain: domain,
			}.New(ak, sk, 0, bucket, ""),
		},
		Key:  key,
		Gzip: true,
	}
}
