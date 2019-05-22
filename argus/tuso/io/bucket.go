package io

import (
	"bufio"
	"bytes"
	"context"
	"fmt"
	"io"
	"net/http"
	"strings"

	"github.com/pkg/errors"
	"github.com/qiniu/rpc.v1"
	"qiniupkg.com/api.v7/kodo"
	"qiniupkg.com/api.v7/kodocli"
	xlog "qiniupkg.com/x/xlog.v7"
)

var _ Blocks = Bucket{}

type BucketConfig struct {
	Ak     string   `json:"ak"`
	Sk     string   `json:"sk"`
	Bucket string   `json:"bucket"`
	Region int      `json:"region"`
	Domain string   `json:"domain"`
	Prefix string   `json:"prefix"`
	IoHost []string `json:"io_host"`
	// kodo.Config
	Zone      int `json:"zone"`
	BlockSize int `json:"block_size"`
}

type Bucket struct {
	BucketConfig
	hub       string
	version   int
	chunkSize int
}

func NewBucket(config BucketConfig, hub string, version int, chunkSize int) Bucket {
	return Bucket{BucketConfig: config, hub: hub, version: version, chunkSize: chunkSize}
}

func (b Bucket) NewWriter(ctx context.Context, offset int) BlockWriter {
	return newBucketWriter(ctx, b, offset)
}

func (b Bucket) NewScanner(ctx context.Context, offset, limit int) BlockScanner {
	return newBucketScanner(ctx, b, offset, limit)
}

func (b Bucket) genKey(ctx context.Context, index int) string {
	return fmt.Sprintf("%s/%s/%d/%d", b.Prefix, b.hub, b.version, index)
}

func (b Bucket) upKey(
	ctx context.Context, client *kodo.Client,
	key string, r io.Reader, size int64,
) error {

	policy := &kodo.PutPolicy{
		Scope:   b.Bucket,
		Expires: 3600,
	}
	token := client.MakeUptoken(policy)
	var uploadConfig = &kodocli.UploadConfig{
	// UpHosts: b.UpHosts,
	// APIHost: b.APIHost,
	}
	uploader := kodocli.NewUploader(b.Zone, uploadConfig)
	var ret = struct {
		Hash string `json:"hash"`
		Key  string `json:"key"`
	}{}
	err := uploader.Put(ctx, &ret, token, key, r, size, nil)
	return err
}

func (b Bucket) dnKey(ctx context.Context, key string) (io.ReadCloser, error) {
	_url := kodo.New(0, &kodo.Config{AccessKey: b.Ak, SecretKey: b.Sk}).MakePrivateUrl(fmt.Sprintf("http://%s/%s", b.Domain, key), nil)
	xl := xlog.New("io.bucket")
	xl.Info("get", _url)

	req, err := http.NewRequest("GET", _url, nil)
	if err != nil {
		return nil, err
	}
	req.URL.Host = strings.TrimPrefix(b.BucketConfig.IoHost[0], "http://")
	// TODO: timeout?
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return nil, errors.Wrap(err, "dnKey http.Do")
	}
	if resp.StatusCode/100 == 2 {
		return resp.Body, nil
	}
	defer resp.Body.Close()
	return nil, errors.Wrap(rpc.ResponseError(resp), "dnKey io error")
}

var _ BlockWriter = &BucketWriter{}

type BucketWriter struct {
	Bucket

	client *kodo.Client

	index  int
	offset int
	buffer []byte

	err error
}

func newBucketWriter(ctx context.Context, bucket Bucket, offset int) *BucketWriter {
	var index = offset * bucket.chunkSize / bucket.BlockSize
	offset = offset * bucket.chunkSize % bucket.BlockSize

	w := &BucketWriter{
		Bucket: bucket, index: index, offset: offset,
		buffer: make([]byte, bucket.BlockSize),
	}

	if offset > 0 {
		func() {
			reader, err := w.Bucket.dnKey(ctx, w.genKey(ctx, w.index))
			if err != nil {
				w.err = err
				return
			}
			defer reader.Close()

			_, err = io.ReadAtLeast(reader, w.buffer, w.offset)
			if err != nil {
				w.err = err
			}
		}()
	}

	return w
}

func (w *BucketWriter) Write(ctx context.Context, bs []byte) error {
	if w.err != nil {
		return w.err
	}
	copy(w.buffer[w.offset:], bs) // TODO check
	w.offset += len(bs)
	if w.offset >= len(w.buffer) {
		err := w.Bucket.upKey(ctx, w.client,
			w.genKey(ctx, w.index),
			bytes.NewBuffer(w.buffer), int64(len(w.buffer)),
		)
		if err != nil {
			return err
		}
		w.index++
		w.offset = 0
	}
	return nil
}

func (w *BucketWriter) Close(ctx context.Context) error {
	if w.offset > 0 {
		err := w.Bucket.upKey(ctx, w.client,
			w.genKey(ctx, w.index),
			bytes.NewBuffer(w.buffer[:w.offset]), int64(w.offset),
		)
		if err != nil {
			return err
		}
	}
	return nil
}

//----------------------------------------------------------------------------//

var _ BlockScanner = &BucketScanner{}

type BucketScanner struct {
	Bucket

	offset_current       int
	offset_end           int
	index                int
	block_offset_current int

	reader    io.ReadCloser
	bufReader *bufio.Reader
	buffer    []byte

	err error
}

func newBucketScanner(ctx context.Context, bucket Bucket, offset, limit int) *BucketScanner {
	var index = offset * bucket.chunkSize / bucket.BlockSize
	offset = offset * bucket.chunkSize % bucket.BlockSize

	return &BucketScanner{
		Bucket:               bucket,
		offset_current:       offset * bucket.chunkSize,
		offset_end:           (offset + limit) * bucket.chunkSize,
		index:                index,
		block_offset_current: offset,
		buffer:               make([]byte, bucket.chunkSize),
	}
}

func (s *BucketScanner) Scan(ctx context.Context) bool {
	if s.offset_current >= s.offset_end {
		return false
	}
	if s.bufReader == nil {
		reader, err := s.Bucket.dnKey(ctx, s.genKey(ctx, s.index))
		if err != nil {
			s.err = err
			return false
		}
		s.reader = reader
		s.bufReader = bufio.NewReaderSize(s.reader, s.Bucket.BlockSize)
	}

	n, err := io.ReadAtLeast(s.bufReader, s.buffer, s.Bucket.chunkSize)
	if err != nil && err != io.EOF {
		s.err = err
		return false
	}

	s.offset_current += s.chunkSize
	s.block_offset_current += s.chunkSize
	if s.block_offset_current >= s.BlockSize {
		s.index++
		s.block_offset_current = 0
		s.bufReader = nil
		return true
	}

	return n == s.chunkSize
}

func (s *BucketScanner) Bytes(ctx context.Context) []byte { return s.buffer[:s.chunkSize] }
func (s *BucketScanner) Error(ctx context.Context) error  { return s.err }
func (s *BucketScanner) Close() error {
	if s.reader != nil {
		return s.reader.Close()
	}
	return nil
}
