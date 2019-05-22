package cc

import (
	"github.com/qiniu/bytes"
	"io"
)

// ---------------------------------------------------

type ReadWriterAt interface {
	io.ReaderAt
	io.WriterAt
}

// ---------------------------------------------------

type Writer struct {
	io.WriterAt
	Offset int64
}

func (p *Writer) Write(val []byte) (n int, err error) {
	n, err = p.WriteAt(val, p.Offset)
	p.Offset += int64(n)
	return
}

// ---------------------------------------------------

type Reader struct {
	io.ReaderAt
	Offset int64
}

func (p *Reader) Read(val []byte) (n int, err error) {
	n, err = p.ReadAt(val, p.Offset)
	p.Offset += int64(n)
	return
}

// ---------------------------------------------------

type NilReader struct{}
type NilWriter struct{}

func (r NilReader) Read(val []byte) (n int, err error) {
	return 0, io.EOF
}

func (r NilWriter) Write(val []byte) (n int, err error) {
	return len(val), nil
}

// ---------------------------------------------------

func NewBytesReader(val []byte) *bytes.Reader {
	return bytes.NewReader(val)
}

// ---------------------------------------------------

func NewBytesWriter(buff []byte) *bytes.Writer {
	return bytes.NewWriter(buff)
}

// ---------------------------------------------------

type optimisticMultiWriter struct {
	writers []io.Writer
	errs    []error
	fail    int
}

func OptimisticMultiWriter(writers ...io.Writer) *optimisticMultiWriter {
	return &optimisticMultiWriter{
		writers: writers,
		errs:    make([]error, len(writers)),
		fail:    0,
	}
}

func (t *optimisticMultiWriter) Write(p []byte) (n int, err error) {

	for i, w := range t.writers {
		if t.errs[i] != nil {
			continue
		}

		_, err1 := w.Write(p)
		if err1 != nil {
			t.fail++
			t.errs[i] = err1
		}
	}

	if t.fail == len(t.writers) {
		return 0, io.ErrShortWrite
	}

	return len(p), nil
}

func (t *optimisticMultiWriter) Errors() []error {

	return t.errs
}

// ---------------------------------------------------
