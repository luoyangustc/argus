package sts

import (
	"context"
	"crypto/sha1"
	"encoding/binary"
	"encoding/hex"
	"io"
	"os"
	"path/filepath"

	"github.com/qiniu/xlog.v1"
	URI "qiniu.com/argus/sts/uri"
)

const (
	_FilePerm os.FileMode = 0755
)

// Disk ...
type Disk interface {
	OpenFileAsReader(ctx context.Context, uri URI.Uri, length *int64) (io.ReadCloser, int64, error)
	OpenFileAsWriter(ctx context.Context, uri URI.Uri, length *int64) (io.WriteCloser, error)
	RemoveAll(ctx context.Context) error
}

var _ Disk = &disk{}

type disk struct {
	dir string
}

func newDisk(ctx context.Context, dir string) (disk, error) {
	return disk{dir: dir}, nil
}

func (d disk) filename(uri URI.Uri) string {
	sum := sha1.Sum([]byte(uri))
	name := hex.EncodeToString(sum[:])
	return filepath.Join(d.dir, name[:3], name[3:6], name[6:])
}

func (d disk) length2Binary(length int64) []byte {
	var bs = make([]byte, 8)
	binary.LittleEndian.PutUint64(bs, uint64(length))
	return bs
}

func (d disk) binary2Length(bs []byte) int64 {
	return int64(binary.LittleEndian.Uint64(bs))
}

func (d disk) OpenFileAsReader(
	ctx context.Context, uri URI.Uri, length *int64,
) (reader io.ReadCloser, length2 int64, err error) {

	xl := xlog.FromContextSafe(ctx)
	name := d.filename(uri)
	xl.Infof("read disk file. %s", name)
	info, err := os.Lstat(name)
	if err != nil {
		if os.IsNotExist(err) {
			return nil, 0, ErrNotExist
		}
		return
	}
	if length != nil && info.Size() != *length+8 {
		return nil, 0, ErrNotExist
	}
	reader, err = os.Open(name)
	if err != nil {
		if os.IsNotExist(err) {
			err = ErrNotExist
		}
		return
	}
	{
		var bs = make([]byte, 8)
		io.ReadFull(reader, bs)
		var length3 = d.binary2Length(bs)
		if length == nil {
			if length3 > 0 {
				if length3+8 != info.Size() {
					reader.Close()
					return nil, 0, ErrNotExist
				}
				length2 = length3
			} else {
				length2 = info.Size() - 8
			}
		} else {
			length2 = *length
		}
	}
	return
}

func (d disk) OpenFileAsWriter(
	ctx context.Context, uri URI.Uri, length *int64,
) (io.WriteCloser, error) {

	xl := xlog.FromContextSafe(ctx)

	name := d.filename(uri)
	if err := os.MkdirAll(filepath.Dir(name), _FilePerm); err != nil && !os.IsExist(err) {
		xl.Errorf("mkdir failed. %s %d %v", filepath.Dir(name), _FilePerm, err)
		return nil, ErrIO
	}
	writer, err := os.OpenFile(name, os.O_RDWR|os.O_CREATE, _FilePerm)
	if err != nil {
		xl.Errorf("open file failed. %s %v", name, err)
		return writer, ErrIO
	}
	var _length int64
	if length != nil {
		_length = *length
	}
	_, err = writer.Write(d.length2Binary(_length))
	if err != nil {
		xl.Errorf("write file head failed. %v", err)
		err = ErrIO
	}
	return writer, err
}

func (d disk) RemoveAll(ctx context.Context) error {
	return os.RemoveAll(d.dir)
}
