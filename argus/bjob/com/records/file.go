package records

import (
	"bufio"
	"bytes"
	"compress/gzip"
	"context"
	"fmt"
	"io"
	"time"

	"github.com/qiniu/xlog.v1"
)

type FileStorage interface {
	List(context.Context) ([]string, error)
	Save(context.Context, string, io.Reader, int64) (string, error)
	Read(context.Context, string) (io.ReadCloser, error)
}

var _ RecordStorage = &File{}
var _ StorageList = &File{}

type File struct {
	FileStorage

	MaxFlushDuration time.Duration
	MaxFlushSize     uint64

	names []string
	buf   *fileBuffer

	now    func() Time
	beginT Time
	count  uint64
}

func NewFile(stg FileStorage, maxFlushDuration time.Duration, maxFlushSize uint64) *File {
	return &File{
		FileStorage:      stg,
		MaxFlushDuration: maxFlushDuration,
		MaxFlushSize:     maxFlushSize,
		now: func() Time {
			_, offset := time.Now().Zone()
			return _Time{
				Duration: maxFlushDuration,
				Time:     time.Now(),
				now:      func() time.Time { return time.Now() },
				offset:   time.Hour * time.Duration(offset),
			}
		},
	}
}

func (f *File) List(context.Context) ([]string, error) { return f.names, nil }

func (f *File) Append(ctx context.Context, key RecordKey, value RecordValue) error {

	var xl = xlog.FromContextSafe(ctx)

	if f.buf == nil {
		f.buf = newFileBuffer()
		f.beginT = f.now()
		f.count = 0
	}
	_ = f.buf.Append(ctx, key, value)
	f.count++

	if f.MaxFlushDuration > 0 || f.MaxFlushSize > 0 {
		if f.MaxFlushDuration > 0 && f.beginT.Before(f.MaxFlushDuration) {
			name := f.genName(ctx, f.now())
			reader, length, _ := f.buf.Flush(ctx)
			key, err := f.Save(ctx, name, reader, length)
			xl.Infof("SAVE: %s %d %v", key, length, err)
			f.names = append(f.names, key)
			f.buf = nil
		} else if f.MaxFlushSize > 0 && f.count >= f.MaxFlushSize {
			if f.now().Get().Sub(f.beginT.Get()) <= time.Second {
				return nil
			}
			name := f.genName(ctx, f.now())
			reader, length, _ := f.buf.Flush(ctx)
			key, err := f.Save(ctx, name, reader, length)
			xl.Infof("SAVE: %s %d %v", key, length, err)
			f.names = append(f.names, key)
			f.buf = nil
		}
	}

	return nil

}

func (f *File) genName(ctx context.Context, endT Time) string {
	return fmt.Sprintf("%s_%s__%d", f.beginT.Format(), endT.Format(), f.count)
}

func (f *File) Flush(ctx context.Context) (string, error) {
	var xl = xlog.FromContextSafe(ctx)
	if f.buf != nil {
		reader, length, _ := f.buf.Flush(ctx)
		name := f.genName(ctx, f.now())
		key, err := f.Save(ctx, name, reader, length)
		xl.Infof("SAVE: %s %d %+v", name, length, err)
		f.buf = nil
		return key, err
	}
	return "", nil
}

func (f *File) Scan(ctx context.Context) _RecordStorageScanner {
	return newFileScanner(ctx, f.FileStorage)
}

var _ _RecordStorageScanner = &FileScanner{}
var _ StorageList = &FileScanner{}

type FileScanner struct {
	FileStorage

	keys   []string
	curkey string

	current int
	rc      io.ReadCloser
	zr      *gzip.Reader
	sc      *bufio.Scanner
}

func newFileScanner(ctx context.Context, stg FileStorage) *FileScanner {
	keys, _ := stg.List(ctx)
	return &FileScanner{FileStorage: stg, keys: keys}
}

func (f *FileScanner) Scan(ctx context.Context) (RecordKey, RecordValue, error) {
	if f.zr == nil {
		if f.current >= len(f.keys) {
			return "", nil, io.EOF
		}

		f.curkey = f.keys[f.current]
		rc, err := f.Read(ctx, f.curkey)
		if err != nil {
			return "", nil, err
		}

		f.rc = rc
		f.zr, _ = gzip.NewReader(f.rc)
		f.sc = bufio.NewScanner(f.zr)
		f.current++
	}

	if !f.sc.Scan() {
		f.sc = nil
		f.zr.Close()
		f.zr = nil
		f.rc.Close()
		f.rc = nil

		return f.Scan(ctx)
	}

	bs := bytes.SplitN(f.sc.Bytes(), []byte("\t"), 2)
	if len(bs) < 2 {
		return "", nil, nil
	}

	return RecordKey(bs[0]), bs[1], nil
}

func (f *FileScanner) Close(context.Context) error {
	if f.zr != nil {
		f.zr.Close()
	}
	if f.rc != nil {
		f.rc.Close()
	}
	return nil
}

func (f *FileScanner) List(ctx context.Context) ([]string, error) { return f.keys, nil }

//----------------------------------------------------------------------------//

type fileBuffer struct {
	buf *bytes.Buffer
	zw  *gzip.Writer
}

func newFileBuffer() *fileBuffer {
	buf := bytes.NewBuffer(nil)
	return &fileBuffer{
		buf: buf,
		zw:  gzip.NewWriter(buf),
	}
}

func (fb *fileBuffer) Append(ctx context.Context, key RecordKey, value RecordValue) error {
	fb.zw.Write([]byte(key))
	fb.zw.Write([]byte("\t"))
	fb.zw.Write([]byte(value))
	fb.zw.Write([]byte("\n"))
	return nil
}

func (fb *fileBuffer) Flush(ctx context.Context) (io.Reader, int64, error) {
	_ = fb.zw.Close()
	return fb.buf, int64(fb.buf.Len()), nil
}

//----------------------------------------------------------------------------//

type Time interface {
	Now() Time
	Before(time.Duration) bool
	Format() string
	Get() time.Time
}

var _ Time = _Time{}

type _Time struct {
	time.Duration
	time.Time
	now    func() time.Time // FOR UT
	offset time.Duration
}

func (t _Time) Format() string { return t.Time.Format("20060102T150405") }
func (t _Time) Get() time.Time { return t.Time }
func (t _Time) Now() Time {
	return _Time{
		Duration: t.Duration,
		Time:     t.now(),
		now:      t.now,
		offset:   t.offset,
	}
}
func (t _Time) Before(d time.Duration) bool {
	now := t.now()
	return !now.Add(-1 * t.offset).Truncate(t.Duration).
		Before(
			t.Time.Add(-1 * t.offset).Truncate(t.Duration).
				Add(d),
		)
}
