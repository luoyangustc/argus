package bucket

import (
	"context"
	"io"
	"time"

	xlog "github.com/qiniu/xlog.v1"
	"qiniupkg.com/api.v7/kodo"
)

type IScanner interface {
	Scan(context.Context, int) (IIter, error)
	Count(context.Context) (int, error)
}

type KeyItem struct {
	Key      string `json:"key"`
	PutTime  int64  `json:"putTime"`
	MimeType string `json:"mimeType"`
}

type IIter interface {
	Next(ctx context.Context) (item KeyItem, beginMarker string, success bool)
	Error() error
}

////////////////////////////////////////////////////////////////////////////////

type Config struct {
	kodo.Config `json:"kodo"`
	Zone        int    `json:"zone"`
	Bucket      string `json:"bucket"`
	Prefix      string `json:"prefix"`
	Domain      string `json:"domain"`
}

func (conf Config) New(ak, sk string, zone int, bucket, prefix string) Config {
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
		Zone:   zone,
		Bucket: bucket,
		Prefix: prefix,
		Domain: conf.Domain,
	}
}

var _ IScanner = Scanner{}

type Scanner struct {
	Config
}

var _ IIter = &EntryIter{}

type EntryIter struct {
	listFunc func(context.Context, string, int) ([]kodo.ListItem, string, error)

	limit      int
	marker     string // 当前截止的marker
	lastMarker string // 上次截止的marker

	count int
	end   bool
	err   error

	items []kodo.ListItem
	index int
}

func (iter *EntryIter) Next(ctx context.Context) (
	item KeyItem, beginMarker string, success bool) {
	if iter.index < len(iter.items) {
		index := iter.index
		iter.index++
		iter.count++
		item = KeyItem{Key: iter.items[index].Key, PutTime: iter.items[index].PutTime, MimeType: iter.items[index].MimeType}
		beginMarker = iter.lastMarker
		success = true
		return
	}
	if iter.end {
		item = KeyItem{}
		beginMarker = iter.lastMarker
		success = false
		return
	}
	entries, marker, err := iter.listFunc(ctx, iter.marker, iter.limit)
	if err != nil {
		iter.end = true
		if err != io.EOF {
			iter.err = err
		}
	}
	iter.lastMarker = iter.marker
	iter.marker = marker
	if entries == nil || len(entries) == 0 {
		item = KeyItem{}
		beginMarker = iter.lastMarker
		success = false
		return
	}
	// iter.count += len(entries)
	iter.count++
	iter.items = entries
	iter.index = 1

	item = KeyItem{Key: iter.items[0].Key, PutTime: iter.items[0].PutTime, MimeType: iter.items[0].MimeType}
	beginMarker = iter.lastMarker
	success = true
	return
}

func (iter *EntryIter) Error() error { return iter.err }

func (b Scanner) Scan(ctx context.Context, bufferSize int) (IIter, error) {
	cli := kodo.New(b.Zone, &b.Config.Config)
	bucket, _ := cli.BucketWithSafe(b.Bucket)
	return &EntryIter{
		listFunc: func(
			ctx context.Context, marker string, limit int,
		) ([]kodo.ListItem, string, error) {
			var (
				xl     = xlog.FromContextSafe(ctx)
				beginT = time.Now()
			)
			entries, _, marker, err := bucket.List(
				ctx, b.Config.Prefix, "", marker, limit)
			if err != nil {
				xl.Warnf(
					"Bucket List Failed. B<%s> P<%s> %d|%d %v",
					b.Config.Bucket, b.Config.Prefix, len(entries), limit, err)
				return entries, marker, err
			}
			xl.Infof("Bucket List: B<%s> P<%s> %d|%d %v %s",
				b.Config.Bucket, b.Config.Prefix,
				len(entries), limit, time.Since(beginT), marker,
			)
			return entries, marker, nil
		},
		limit: bufferSize,
	}, nil
}

func (b Scanner) Count(ctx context.Context) (int, error) {
	iter, _ := b.Scan(ctx, 500)
	for {
		_, _, ok := iter.Next(ctx)
		if !ok {
			break
		}
	}
	return iter.(*EntryIter).count, iter.Error()
}
