package io

import (
	"context"
	"io"
)

type Blocks interface {
	NewWriter(ctx context.Context, offset int) BlockWriter
	NewScanner(ctx context.Context, offset, limit int) BlockScanner
}

type BlockWriter interface {
	Write(ctx context.Context, bs []byte) error
	Close(ctx context.Context) error
}

type BlockScanner interface {
	Scan(ctx context.Context) bool
	Bytes(ctx context.Context) []byte
	Error(ctx context.Context) error
	io.Closer
}

////////////////////////////////////////////////////////////////////////////////

type MockBlocks struct {
	bytes                []byte
	chunkSize, blockSize int
	offset               int
}

func NewMockBlocks(chunkSize, size int) Blocks {
	return &MockBlocks{
		bytes:     make([]byte, chunkSize*size),
		chunkSize: chunkSize,
		blockSize: size,
		offset:    -1,
	}
}

func (bs *MockBlocks) NewWriter(ctx context.Context, offset int) BlockWriter {
	if (offset+1)*bs.chunkSize > len(bs.bytes) {
		blocks := offset/bs.blockSize + 1
		_bs := make([]byte, blocks*bs.blockSize)
		copy(_bs, bs.bytes)
		bs.bytes = _bs
	}
	return &MockBlockWriter{MockBlocks: bs, offset: offset}
}
func (bs *MockBlocks) NewScanner(ctx context.Context, offset, limit int) BlockScanner {
	var end = offset + limit
	if limit <= 0 {
		end = -1
	}
	return &MockBlockScanner{MockBlocks: bs, offset: offset, end: end}
}

type MockBlockWriter struct {
	*MockBlocks
	offset int
}

func (w *MockBlockWriter) Write(ctx context.Context, bs []byte) error {
	if (w.offset+1)*w.chunkSize > len(w.bytes) {
		bs := make([]byte, len(w.bytes)+w.chunkSize*w.blockSize)
		copy(bs, w.bytes)
		w.bytes = bs
	}
	copy(w.bytes[w.offset*w.chunkSize:], bs)
	w.offset += 1
	w.MockBlocks.offset = w.offset
	return nil
}

func (w *MockBlockWriter) Close(ctx context.Context) error { return nil }

type MockBlockScanner struct {
	*MockBlocks
	offset int
	end    int
}

func (s *MockBlockScanner) Scan(ctx context.Context) bool {
	if (s.end > 0 && s.offset >= s.end) || s.offset >= s.MockBlocks.offset {
		return false
	}
	return true
}

func (s *MockBlockScanner) Bytes(ctx context.Context) []byte {
	var offset = s.offset
	s.offset = offset + 1
	return s.bytes[offset*s.chunkSize : (offset+1)*s.chunkSize]
}

func (s *MockBlockScanner) Error(ctx context.Context) error { return nil }
func (s *MockBlockScanner) Close() error                    { return nil }

////////////////////////////////////////////////////////////////////////////////
