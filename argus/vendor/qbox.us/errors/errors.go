package errors

import (
	"errors"
	"io"
	"syscall"
)

// --------------------------------------------------------------------
// 系统错误

var EACCES = syscall.EACCES
var EEXIST = syscall.EEXIST
var ENOENT = syscall.ENOENT
var EINVAL = syscall.EINVAL
var ENOTDIR = syscall.ENOTDIR
var ENFILE = syscall.ENFILE
var EIO = syscall.EIO
var ENOSPC = syscall.ENOSPC

var ErrNoSpace = ENOSPC

var EOF = io.EOF
var ErrUnexpectedEOF = io.ErrUnexpectedEOF
var ErrShortWrite = io.ErrShortWrite
var ErrShortBuffer = io.ErrShortBuffer
var ErrClosedPipe = io.ErrClosedPipe
var ErrUnmatchedChecksum = errors.New("unmatched checksum")

var g_errtbl = map[string]error{

	EACCES.Error():  EACCES,
	EEXIST.Error():  EEXIST,
	ENOENT.Error():  ENOENT,
	EINVAL.Error():  EINVAL,
	ENOTDIR.Error(): ENOTDIR,
	ENFILE.Error():  ENFILE,
	EIO.Error():     EIO,
	ENOSPC.Error():  ENOSPC,

	EOF.Error():                  EOF,
	ErrUnexpectedEOF.Error():     ErrUnexpectedEOF,
	ErrShortWrite.Error():        ErrShortWrite,
	ErrShortBuffer.Error():       ErrShortBuffer,
	ErrClosedPipe.Error():        ErrClosedPipe,
	ErrUnmatchedChecksum.Error(): ErrUnmatchedChecksum,
}

func New(err string) error {
	if e, ok := g_errtbl[err]; ok {
		return e
	}
	return errors.New(err)
}

func Register(err string) error {
	if e, ok := g_errtbl[err]; ok {
		return e
	}
	e := errors.New(err)
	g_errtbl[err] = e
	return e
}

func RegisterError(err error) error {
	s := err.Error()
	if e, ok := g_errtbl[s]; ok {
		return e
	}
	g_errtbl[s] = err
	return err
}

// --------------------------------------------------------------------
