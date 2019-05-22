package log

import (
	"github.com/qiniu/filelog"
)

// ----------------------------------------------------------

type Logger struct {
	w *filelog.Writer
}

func NewLogger(dir, prefix string, timeMode int64, chunkBits uint) (r *Logger, err error) {
	w, err := filelog.NewWriter(dir, prefix, timeMode, chunkBits)
	if err != nil {
		return
	}
	r = &Logger{w}
	return
}

func (r *Logger) Close() (err error) {
	return r.w.Close()
}

func (r *Logger) Log(msg []byte) (err error) {
	msg = append(msg, '\n')
	_, err = r.w.Write(msg)
	return
}
