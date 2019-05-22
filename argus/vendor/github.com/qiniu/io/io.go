package io

import (
	"io"
)

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

type OptimisticWriter struct {
	io.Writer
	Err     error
	Written int64
}

type OptimisticMultiWriter struct {
	Writers []OptimisticWriter
	Fails   int
}

func (p *OptimisticMultiWriter) Write(buf []byte) (n int, err error) {

	for i, w := range p.Writers {
		if w.Err != nil {
			continue
		}
		n1, err1 := w.Write(buf)
		if err1 != nil {
			p.Writers[i].Err = err1
			p.Fails++
		}
		p.Writers[i].Written += int64(n1)
	}
	return len(buf), nil
}

// ---------------------------------------------------
