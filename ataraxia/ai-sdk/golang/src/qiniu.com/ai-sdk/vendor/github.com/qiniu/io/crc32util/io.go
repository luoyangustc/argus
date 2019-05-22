package crc32util

import (
	"io"
	"io/ioutil"
)

func newSectionReader(r io.Reader, off int64, n int64) io.Reader {

	return &sectionReader{r: r, base: off, n: n}
}

type sectionReader struct {
	r       io.Reader
	base    int64
	n       int64
	discard bool
}

func (s *sectionReader) Read(p []byte) (int, error) {
	if !s.discard {
		_, err := io.CopyN(ioutil.Discard, s.r, s.base)
		if err != nil {
			return 0, err
		}
		s.discard = true
		s.r = io.LimitReader(s.r, s.n)
	}
	return s.r.Read(p)
}
