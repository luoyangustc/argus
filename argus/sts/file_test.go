package sts

import (
	"bytes"
	"context"
	"io"
	"io/ioutil"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"

	URI "qiniu.com/argus/sts/uri"
)

func TestFile(t *testing.T) {

	disk := newMemory()

	uri := URI.NewURI(time.Now().Format("20060102150405"))
	bs := []byte("xxxxxxxxxxxxxxxxx")
	length := int64(len(bs))

	d := newFile(uri, &length, disk, nil)
	func() {
		_, err := d.OpenAsReader(context.Background())
		assert.Equal(t, err, ErrNotExist)
	}()
	func() {
		_, err := d.Write(context.Background(), bytes.NewBuffer(bs))
		assert.NoError(t, err, "write file")
	}()
	func() {
		reader, err := d.OpenAsReader(context.Background())
		assert.NoError(t, err, "open file as reader")
		defer reader.Close()
		bs2, _ := ioutil.ReadAll(reader)
		assert.Equal(t, len(bs), len(bs2))
	}()
	func() {
		d.length = nil
		reader, err := d.OpenAsReader(context.Background())
		assert.NoError(t, err, "open file as reader")
		defer reader.Close()
		bs2, _ := ioutil.ReadAll(reader)
		assert.Equal(t, len(bs), len(bs2))
		assert.Equal(t, len(bs), int(*d.length))
	}()
}

func TestFilePipe(t *testing.T) {

	disk := newMemory()

	uri := URI.NewURI(time.Now().Format("20060102150405"))
	bs := []byte("xxxxxxxxxxxxxxxxx")
	length := int64(len(bs))

	d := newFile(uri, &length, disk, nil)
	reader, ch := newSlowReader(bytes.NewBuffer(bs), time.Second*2)
	go func() {
		_, err := d.Write(context.Background(), reader)
		assert.NoError(t, err, "write file")
	}()
	func() {
		select {
		case <-ch:
		}
		reader, err := d.OpenAsReader(context.Background())
		assert.NoError(t, err, "open file as reader")
		defer reader.Close()
		bs2, _ := ioutil.ReadAll(reader)
		assert.Equal(t, len(bs), len(bs2))
	}()

}

func TestFilePipe2(t *testing.T) {

	disk := newMemory()

	uri := URI.NewURI(time.Now().Format("20060102150405"))
	bs := []byte("xxxxxxxxxxxxxxxxx")
	length := int64(len(bs))

	d := newFile(uri, &length, disk, nil)
	reader, ch := newSlowReader(bytes.NewBuffer(bs), time.Second*2)
	go func() {
		writer, err := d.OpenAsWriter(context.Background(), &length)
		assert.NoError(t, err, "open write file")
		defer writer.Close()
		_, err = io.Copy(writer, reader)
		assert.NoError(t, err, "copy")
	}()
	func() {
		select {
		case <-ch:
		}
		reader, err := d.OpenAsReader(context.Background())
		assert.NoError(t, err, "open file as reader")
		defer reader.Close()
		bs2, _ := ioutil.ReadAll(reader)
		assert.Equal(t, len(bs), len(bs2))
	}()

}

type slowReader struct {
	io.Reader
	duration time.Duration

	beginToRead chan bool
}

func newSlowReader(reader io.Reader, duration time.Duration) (*slowReader, chan bool) {
	ch := make(chan bool)
	return &slowReader{
		Reader:      reader,
		duration:    duration,
		beginToRead: ch,
	}, ch
}

func (r *slowReader) Read(p []byte) (n int, err error) {
	if r.beginToRead != nil {
		close(r.beginToRead)
		r.beginToRead = nil
	}
	time.Sleep(r.duration)
	return r.Reader.Read(p)
}

func (r *slowReader) Close() error { return nil }
