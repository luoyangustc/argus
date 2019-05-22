package gpu

import (
	"errors"
)

var (
	// cuda error
	ErrWriteCudaBuffer = errors.New("write to cuda buffer error")
	ErrSliceBuffer     = errors.New("fail to slice cuda buffer")
	ErrClearCudaBuffer = errors.New("clear cuda buffer failed")
	ErrAllDevice       = errors.New("fail to list all cuda device")
	ErrCreateContext   = errors.New("fail to create cuda context")
)
