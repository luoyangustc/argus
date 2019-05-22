package log

import (
	"github.com/qiniu/largefile"
	"sync/atomic"
)

// ----------------------------------------------------------

type Logger struct {
	off int64
	f   *largefile.File
}

func New(f *largefile.File) (r *Logger, err error) {
	fsize, err := f.FsizeOf()
	if err != nil {
		return
	}
	return &Logger{fsize, f}, nil
}

func Open(name string, chunkBits uint) (r *Logger, err error) {
	f, err := largefile.Open(name, chunkBits)
	if err != nil {
		return
	}
	return New(f)
}

func (r *Logger) Close() (err error) {
	return r.f.Close()
}

func (r *Logger) Log(msg []byte) (err error) {
	msg = append(msg, '\n')
	n := int64(len(msg))
	off := atomic.AddInt64(&r.off, n)
	_, err = r.f.WriteAt(msg, off-n)
	return
}

// ----------------------------------------------------------
