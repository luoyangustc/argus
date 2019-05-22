package search

import (
	"errors"
	"net/http"

	"github.com/qiniu/http/httputil.v1"
)

var (
	// server error
	ErrFeatureSetExist    = httputil.NewError(http.StatusBadRequest, "feature set is already exist")
	ErrInvalidDeviceID    = errors.New("invalid device id")
	ErrInvalidFeatures    = httputil.NewError(http.StatusBadRequest, "invalid features in request")
	ErrFeatureSetNotFound = httputil.NewError(http.StatusNotFound, "feature set not found")
	ErrInvalidSetState    = httputil.NewError(http.StatusBadRequest, "invalid set state")
	ErrFeatureNotFound    = errors.New("feature not found")
	ErrUseGPUMode         = errors.New("use gpu mode in cpu programm")

	// cache error
	ErrTooMuchGPUMemory = errors.New("use too much gpu memory")
	ErrAllocatGPUMemory = errors.New("fail to allocate gpu memory")
	ErrSliceGPUBuffer   = errors.New("fail to slice gpu buffer")
	ErrNotEnoughBlocks  = errors.New("cache does not have enough blocks")

	// feature set error
	ErrOutOfBatch              = errors.New("requests out of batch limit")
	ErrMismatchDimension       = errors.New("feature with mismatch dimension")
	ErrWriteInputBuffer        = errors.New("failed to write input buffer")
	ErrWriteOutputBuffer       = errors.New("failed to write output buffer")
	ErrDeleteEmptyID           = errors.New("can not delete empty id")
	ErrInvalidCompareTargetSet = errors.New("invalid compare target set")

	// block error
	ErrBlockIsFull         = errors.New("block is full")
	ErrInvalidFeatureValue = errors.New("invalid feature value")
)
