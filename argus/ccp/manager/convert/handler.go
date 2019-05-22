package convert

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

type Handler struct {
	kodo.Config
}

////////////////////////////////////////////////////////////////////////////////

type LineIter interface {
	Next(ctx context.Context) (string, bool)
	Error() error
	Close() error
}

var _ LineIter = &KeysIter{}

type KeysIter struct {
	BUCKET.Bucket
	Keys []string
	Gzip bool

	i, j    int
	r       io.ReadCloser
	gr      io.ReadCloser
	scanner *bufio.Scanner

	count int
	end   bool
	err   error
}

func (iter *KeysIter) Next(ctx context.Context) (string, bool) {

	xl := xlog.FromContextSafe(ctx)

	if iter.end {
		xl.Info("iter.end")
		return "", false
	}

	if iter.scanner == nil {
		if iter.i >= len(iter.Keys) {
			xl.Infof("iter.i over len, %d", iter.i)
			return "", false
		}

		var err error
		iter.r, err = iter.Bucket.ReadByDomain(ctx, iter.Keys[iter.i])
		xl.Infof("iter.Bucket.Read, %s, %v", iter.Keys[iter.i], err)
		if err != nil {
			iter.end = true
			iter.err = err
			return "", false
		}
		xl.Infof("iter.Gzip, %v", iter.Gzip)
		if iter.Gzip {
			iter.gr, _ = gzip.NewReader(iter.r)
			iter.scanner = bufio.NewScanner(iter.gr)
			iter.scanner.Buffer(make([]byte, 0, 1024*64), 1024*1024*16)
		} else {
			iter.scanner = bufio.NewScanner(iter.r)
			iter.scanner.Buffer(make([]byte, 0, 1024*64), 1024*1024*16)
		}
	}

	if !iter.scanner.Scan() {

		if iter.Gzip {
			iter.gr.Close()
			iter.gr = nil
		}
		iter.r.Close()
		iter.r = nil
		iter.j = 0
		iter.i++

		err := iter.scanner.Err()
		xl.Infof("iter.scanner.Err = %v", err)
		if err != nil {
			iter.end = true
			iter.err = err
			iter.scanner = nil
			return "", false
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
	return line, true
}

func (iter *KeysIter) Error() error { return iter.err }
func (iter *KeysIter) Close() error { return nil }

func (h Handler) ScannerBjobResult(
	ctx context.Context,
	ak, sk string, bucket, domain string,
	keys []string,
) *KeysIter {

	xl := xlog.FromContextSafe(ctx)

	xl.Infof("Scanner, %s, %s, %v", bucket, domain, keys)

	return &KeysIter{
		Bucket: BUCKET.Bucket{
			Config: BUCKET.Config{
				Config: h.Config,
				Bucket: bucket,
				Domain: domain,
			}.New(ak, sk, 0, bucket, ""),
		},
		Keys: keys,
		Gzip: true,
	}

}
