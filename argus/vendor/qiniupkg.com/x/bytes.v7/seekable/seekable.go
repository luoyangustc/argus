// This package provide a method to read and replace http.Request's body.
package seekable

import (
	"errors"
	"io"
	"io/ioutil"
	"net/http"

	"qiniupkg.com/x/bytes.v7"
)

// ---------------------------------------------------

type Seekabler interface {
	Bytes() []byte
	Read(val []byte) (n int, err error)
	SeekToBegin() error
}

type SeekableCloser interface {
	Seekabler
	io.Closer
}

// ---------------------------------------------------

type readCloser struct {
	Seekabler
	io.Closer
}

var ErrNoBody = errors.New("no body")
var ErrTooLargeBody = errors.New("too large body")

var MaxBodyLength int64 = 16 * 1024 * 1024

func New(req *http.Request) (r SeekableCloser, err error) {
	if req.Body == nil {
		return nil, ErrNoBody
	}
	var ok bool
	if r, ok = req.Body.(SeekableCloser); ok {
		return
	}
	b, err2 := ReadAll(req)
	if err2 != nil {
		return nil, err2
	}
	r = bytes.NewReader(b)
	req.Body = readCloser{r, req.Body}
	return
}

type readCloser2 struct {
	io.Reader
	io.Closer
}

func ReadAll(req *http.Request) (b []byte, err error) {
	if req.ContentLength > MaxBodyLength {
		return nil, ErrTooLargeBody
	} else if req.ContentLength > 0 {
		b = make([]byte, int(req.ContentLength))
		_, err = io.ReadFull(req.Body, b)
		return
	} else if req.ContentLength == 0 {
		return nil, ErrNoBody
	}
	b, err = ioutil.ReadAll(io.LimitReader(req.Body, MaxBodyLength+1))
	if int64(len(b)) > MaxBodyLength {
		r := io.MultiReader(bytes.NewReader(b), req.Body)
		req.Body = readCloser2{r, req.Body}
		return nil, ErrTooLargeBody
	}
	return
}

// ---------------------------------------------------
