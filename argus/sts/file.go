package sts

import (
	"context"
	"io"
	"sync"
	"sync/atomic"
	"time"

	"github.com/qiniu/xlog.v1"
	URI "qiniu.com/argus/sts/uri"
)

// File ...
type File interface {
	OpenAsReader(ctx context.Context) (io.ReadCloser, error)
	Write(ctx context.Context, reader io.Reader) (int64, error)
	OpenAsWriter(ctx context.Context, length *int64) (io.WriteCloser, error)

	Acquire()
	Release()

	SetLength(int64)
	GetLength() int64
}

var _ File = &file{}

//
// DESC: 当前主要为小文件（<=4MB)IO，暂时以顺序`写完再读`的逻辑简化实现
//
type file struct {
	uri    URI.Uri
	length *int64
	disk   Disk
	ref    int32 // Exist or IO
	rw     *sync.RWMutex

	releaseFunc func(check func() bool)
}

func newFile(uri URI.Uri, length *int64, disk Disk, releaseFunc func(func() bool)) *file {
	return &file{
		uri:    uri,
		length: length,
		disk:   disk,
		rw:     new(sync.RWMutex),

		releaseFunc: releaseFunc,
	}
}

func (f *file) Acquire() { atomic.AddInt32(&f.ref, 1) }

func (f *file) Release() {
	ref := atomic.AddInt32(&f.ref, -1)
	if ref == 0 && f.releaseFunc != nil {
		f.releaseFunc(func() bool { return atomic.LoadInt32(&f.ref) == 0 })
	}
}

func (f *file) SetLength(length int64) {
	if length > 0 {
		f.length = &length
	}
}
func (f *file) GetLength() int64 {
	if f.length == nil {
		return 0
	}
	return *f.length
}

func (f *file) OpenAsReader(ctx context.Context) (io.ReadCloser, error) {
	begin := time.Now()
	waitForLock("file", "", "AsReader").Inc()
	f.rw.RLock()
	responseTime().
		WithLabelValues("file.Wait4Reader", formatError(nil)).
		Observe(float64(time.Since(begin)) / 1e9)
	var length int64
	reader, length, err := f.disk.OpenFileAsReader(ctx, f.uri, f.length)
	if err != nil {
		f.Release()
		f.rw.RUnlock()
		waitForLock("file", "", "AsReader").Dec()
		return nil, err
	}
	f.length = &length
	atomic.AddInt32(&f.ref, 1) // BEGIN OF READ
	return newFileReader(f, reader), nil
}

func (f *file) Write(ctx context.Context, reader io.Reader) (int64, error) {
	xl := xlog.FromContextSafe(ctx)

	waitForLock("file", "", "Write").Inc()
	f.rw.Lock()
	defer waitForLock("file", "", "Write").Dec()
	defer f.rw.Unlock()
	atomic.AddInt32(&f.ref, 1) // BEGIN OF WRITE
	defer f.Release()          // END OF WRITE
	writer, err := f.disk.OpenFileAsWriter(ctx, f.uri, f.length)
	if err != nil {
		xl.Errorf("open file as writer failed. %s %s", f.uri, err)
		return 0, err
	}
	defer writer.Close()
	var n int64
	if f.length != nil {
		n = *f.length
		_, err = io.CopyN(writer, reader, n)
	} else {
		n, err = io.Copy(writer, reader)
	}
	if err != nil {
		atomic.AddInt32(&f.ref, 1) // EXIST
	}
	xl.Infof("copy to file. %s %d %v", f.uri, n, err)
	return n, err
}

func (f *file) OpenAsWriter(ctx context.Context, length *int64) (io.WriteCloser, error) {
	xl := xlog.FromContextSafe(ctx)

	waitForLock("file", "", "AsWriter").Inc()
	f.rw.Lock()
	// defer f.rw.Unlock()
	atomic.AddInt32(&f.ref, 1) // BEGIN OF WRITE
	// defer f.Release()          // END OF WRITE
	writer, err := f.disk.OpenFileAsWriter(ctx, f.uri, f.length)
	if err != nil {
		xl.Errorf("open file as writer failed. %s %s", f.uri, err)
		f.Release()
		f.rw.Unlock()
		waitForLock("file", "", "AsWriter").Dec()
		return nil, err
	}
	return newFileWriter(f, writer, length), nil
}

var _ io.ReadCloser = &fileReader{}

type fileReader struct {
	file   *file
	reader io.ReadCloser
}

func newFileReader(file *file, reader io.ReadCloser) *fileReader {
	return &fileReader{
		file:   file,
		reader: reader,
	}
}

func (r *fileReader) Read(p []byte) (n int, err error) {
	return r.reader.Read(p)
}

func (r *fileReader) Close() error {
	r.file.Release() // END OF READ
	r.file.rw.RUnlock()
	waitForLock("file", "", "AsReader").Dec()
	err := r.reader.Close()
	return err
}

type fileWriter struct {
	file   *file
	writer io.WriteCloser
	length *int64
	err    error
}

func newFileWriter(file *file, writer io.WriteCloser, length *int64) io.WriteCloser {
	return &fileWriter{
		file:   file,
		writer: writer,
		length: length,
	}
}

func (w *fileWriter) Write(p []byte) (n int, err error) {
	n, err = w.writer.Write(p)
	w.err = err
	return
}

func (w *fileWriter) Close() error {
	if w.err != nil {
		atomic.AddInt32(&w.file.ref, 1) // EXIST
	}
	err := w.writer.Close()
	w.file.Release()
	w.file.Release() // For Open
	w.file.rw.Unlock()
	waitForLock("file", "", "AsWriter").Dec()
	return err
}
