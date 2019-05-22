package sts

import (
	"bytes"
	"context"
	"io"
	"io/ioutil"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"

	URI "qiniu.com/argus/sts/uri"
)

func TestFilenameInDisk(t *testing.T) {

	d := disk{dir: "/foo"}
	name := d.filename("xxxx")
	assert.Equal(t, "/foo/4ad/583/af22c2e7d40c1c916b2920299155a46464", name)

}

func TestLengthInDisk(t *testing.T) {
	d := disk{}
	assert.Equal(t, int64(1234), d.binary2Length(d.length2Binary(1234)))
}

func TestDiskIO(t *testing.T) {

	dir := filepath.Join(os.TempDir(), time.Now().Format("20060102150405"))
	d, err := newDisk(context.Background(), dir)
	assert.NoError(t, err, "new disk")

	uri := URI.NewURI(time.Now().Format("20060102150405"))
	bs := []byte("xxxxxxxxxxxxxxxxx")
	length := int64(len(bs))

	func() {
		_, _, err := d.OpenFileAsReader(context.Background(), uri, &length)
		assert.Equal(t, ErrNotExist, err)
	}()

	func() {
		writer, err := d.OpenFileAsWriter(context.Background(), uri, &length)
		assert.NoError(t, err, "open file as writer")
		defer writer.Close()
		_, err = writer.Write(bs)
		assert.NoError(t, err, "write file")
	}()

	func() {
		reader, _, err := d.OpenFileAsReader(context.Background(), uri, &length)
		assert.NoError(t, err, "open file as reader")
		defer reader.Close()
		bs2, _ := ioutil.ReadAll(reader)
		assert.Equal(t, len(bs), len(bs2))
	}()

	func() {
		reader, length2, err := d.OpenFileAsReader(context.Background(), uri, nil)
		assert.NoError(t, err, "open file as reader")
		defer reader.Close()
		bs2, _ := ioutil.ReadAll(reader)
		assert.Equal(t, len(bs), len(bs2))
		assert.Equal(t, len(bs), int(length2))
	}()
}

////////////////////////////////////////////////////////////////////////////////

var _ Disk = &memory{}

type memory struct {
	m map[URI.Uri][]byte
}

func newMemory() *memory {
	return &memory{m: make(map[URI.Uri][]byte)}
}

func (m *memory) OpenFileAsReader(ctx context.Context, uri URI.Uri, length *int64) (io.ReadCloser, int64, error) {
	if bs, ok := m.m[uri]; ok {
		return &memoryReader{buf: bytes.NewBuffer(bs)}, int64(len(bs)), nil
	}
	return nil, 0, ErrNotExist
}

func (m *memory) OpenFileAsWriter(ctx context.Context, uri URI.Uri, length *int64) (io.WriteCloser, error) {
	return &memoryWriter{
		buf:    bytes.NewBuffer(nil),
		uri:    uri,
		length: *length,
		memory: m,
	}, nil
}

func (m *memory) RemoveAll(ctx context.Context) error { return nil }

type memoryReader struct {
	buf *bytes.Buffer
}

func (m *memoryReader) Read(p []byte) (n int, err error) { return m.buf.Read(p) }
func (m *memoryReader) Close() error                     { return nil }

type memoryWriter struct {
	buf    *bytes.Buffer
	uri    URI.Uri
	length int64
	memory *memory
}

func (m *memoryWriter) Write(p []byte) (n int, err error) { return m.buf.Write(p) }

func (m *memoryWriter) Close() error {
	if int64(m.buf.Len()) == m.length {
		m.memory.m[m.uri] = m.buf.Bytes()
	}
	return nil
}
