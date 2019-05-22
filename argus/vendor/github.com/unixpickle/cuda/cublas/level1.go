// +build cublas

package cublas

import (
	"unsafe"

	"github.com/unixpickle/cuda"
)

/*
#include <cublas_v2.h>
*/
import "C"

// Sdot performs a single-precision dot product.
//
// The result argument's type depends on the pointer mode.
// In the Host pointer mode, it should be *float32.
// In the Device pointer mode, it should be a cuda.Buffer.
//
// This must be called inside the cuda.Context.
func (h *Handle) Sdot(n int, x cuda.Buffer, incx int, y cuda.Buffer, incy int,
	result interface{}) error {
	if n < 0 {
		panic("size out of bounds")
	} else if stridedSize(x.Size()/4, incx) < uintptr(n) {
		panic("index out of bounds")
	} else if stridedSize(y.Size()/4, incy) < uintptr(n) {
		panic("index out of bounds")
	}
	var res C.cublasStatus_t
	x.WithPtr(func(xPtr unsafe.Pointer) {
		y.WithPtr(func(yPtr unsafe.Pointer) {
			if h.PointerMode() == Host {
				res = C.cublasSdot(h.handle, safeIntToC(n),
					(*C.float)(xPtr), safeIntToC(incx),
					(*C.float)(yPtr), safeIntToC(incy),
					(*C.float)(result.(*float32)))
			} else {
				b := result.(cuda.Buffer)
				if b.Size() < 4 {
					panic("buffer underflow")
				}
				b.WithPtr(func(outPtr unsafe.Pointer) {
					res = C.cublasSdot(h.handle, safeIntToC(n),
						(*C.float)(xPtr), safeIntToC(incx),
						(*C.float)(yPtr), safeIntToC(incy),
						(*C.float)(outPtr))
				})
			}
		})
	})
	return newError("cublasSdot", res)
}

// Ddot performs a double-precision dot product.
//
// The result argument's type depends on the pointer mode.
// In the Host pointer mode, it should be *float64.
// In the Device pointer mode, it should be a cuda.Buffer.
//
// This must be called inside the cuda.Context.
func (h *Handle) Ddot(n int, x cuda.Buffer, incx int, y cuda.Buffer, incy int,
	result interface{}) error {
	if n < 0 {
		panic("size out of bounds")
	} else if stridedSize(x.Size()/8, incx) < uintptr(n) {
		panic("index out of bounds")
	} else if stridedSize(y.Size()/8, incy) < uintptr(n) {
		panic("index out of bounds")
	}
	var res C.cublasStatus_t
	x.WithPtr(func(xPtr unsafe.Pointer) {
		y.WithPtr(func(yPtr unsafe.Pointer) {
			if h.PointerMode() == Host {
				res = C.cublasDdot(h.handle, safeIntToC(n),
					(*C.double)(xPtr), safeIntToC(incx),
					(*C.double)(yPtr), safeIntToC(incy),
					(*C.double)(result.(*float64)))
			} else {
				b := result.(cuda.Buffer)
				if b.Size() < 8 {
					panic("buffer underflow")
				}
				b.WithPtr(func(outPtr unsafe.Pointer) {
					res = C.cublasDdot(h.handle, safeIntToC(n),
						(*C.double)(xPtr), safeIntToC(incx),
						(*C.double)(yPtr), safeIntToC(incy),
						(*C.double)(outPtr))
				})
			}
		})
	})
	return newError("cublasDdot", res)
}

// Sscal scales a single-precision vector.
//
// The argument alpha's type depends on the pointer mode.
// In the Host pointer mode, use float32 or *float32.
// In the Device pointer mode, use cuda.Buffer.
//
// This must be called inside the cuda.Context.
func (h *Handle) Sscal(n int, alpha interface{}, x cuda.Buffer, incx int) error {
	if n < 0 {
		panic("size out of bounds")
	} else if stridedSize(x.Size()/4, incx) < uintptr(n) {
		panic("index out of bounds")
	}

	var res C.cublasStatus_t
	x.WithPtr(func(ptr unsafe.Pointer) {
		if h.PointerMode() == Host {
			pointerizeInputs(&alpha)
			res = C.cublasSscal(h.handle, safeIntToC(n), (*C.float)(alpha.(*float32)),
				(*C.float)(ptr), safeIntToC(incx))
		} else {
			b := alpha.(cuda.Buffer)
			if b.Size() < 4 {
				panic("buffer underflow")
			}
			b.WithPtr(func(alphaPtr unsafe.Pointer) {
				res = C.cublasSscal(h.handle, safeIntToC(n), (*C.float)(alphaPtr),
					(*C.float)(ptr), safeIntToC(incx))
			})
		}
	})

	return newError("cublasSscal", res)
}

// Dscal is like Sscal, but for double-precision.
//
// The argument alpha's type depends on the pointer mode.
// In the Host pointer mode, use float64 or *float64.
// In the Device pointer mode, use cuda.Buffer.
//
// This must be called inside the cuda.Context.
func (h *Handle) Dscal(n int, alpha interface{}, x cuda.Buffer, incx int) error {
	if n < 0 {
		panic("size out of bounds")
	} else if stridedSize(x.Size()/8, incx) < uintptr(n) {
		panic("index out of bounds")
	}

	var res C.cublasStatus_t
	x.WithPtr(func(ptr unsafe.Pointer) {
		if h.PointerMode() == Host {
			pointerizeInputs(&alpha)
			res = C.cublasDscal(h.handle, safeIntToC(n), (*C.double)(alpha.(*float64)),
				(*C.double)(ptr), safeIntToC(incx))
		} else {
			b := alpha.(cuda.Buffer)
			if b.Size() < 8 {
				panic("buffer underflow")
			}
			b.WithPtr(func(alphaPtr unsafe.Pointer) {
				res = C.cublasDscal(h.handle, safeIntToC(n), (*C.double)(alphaPtr),
					(*C.double)(ptr), safeIntToC(incx))
			})
		}
	})

	return newError("cublasDscal", res)
}

// Saxpy computes single-precision "ax plus y".
//
// The argument alpha's type depends on the pointer mode.
// In the Host pointer mode, use float32 or *float32.
// In the Device pointer mode, use cuda.Buffer.
//
// This must be called inside the cuda.Context.
func (h *Handle) Saxpy(n int, alpha interface{}, x cuda.Buffer, incx int,
	y cuda.Buffer, incy int) error {
	if n < 0 {
		panic("size out of bounds")
	} else if stridedSize(x.Size()/4, incx) < uintptr(n) {
		panic("index out of bounds")
	} else if stridedSize(y.Size()/4, incy) < uintptr(n) {
		panic("index out of bounds")
	}

	var res C.cublasStatus_t
	x.WithPtr(func(xPtr unsafe.Pointer) {
		y.WithPtr(func(yPtr unsafe.Pointer) {
			if h.PointerMode() == Host {
				pointerizeInputs(&alpha)
				res = C.cublasSaxpy(h.handle, safeIntToC(n), (*C.float)(alpha.(*float32)),
					(*C.float)(xPtr), safeIntToC(incx),
					(*C.float)(yPtr), safeIntToC(incy))
			} else {
				b := alpha.(cuda.Buffer)
				if b.Size() < 4 {
					panic("buffer underflow")
				}
				b.WithPtr(func(alphaPtr unsafe.Pointer) {
					res = C.cublasSaxpy(h.handle, safeIntToC(n), (*C.float)(alphaPtr),
						(*C.float)(xPtr), safeIntToC(incx),
						(*C.float)(yPtr), safeIntToC(incy))
				})
			}
		})
	})

	return newError("cublasSaxpy", res)
}

// Daxpy is like Saxpy, but for double-precision.
//
// The argument alpha's type depends on the pointer mode.
// In the Host pointer mode, use float64 or *float64.
// In the Device pointer mode, use cuda.Buffer.
//
// This must be called inside the cuda.Context.
func (h *Handle) Daxpy(n int, alpha interface{}, x cuda.Buffer, incx int,
	y cuda.Buffer, incy int) error {
	if n < 0 {
		panic("size out of bounds")
	} else if stridedSize(x.Size()/8, incx) < uintptr(n) {
		panic("index out of bounds")
	} else if stridedSize(y.Size()/8, incy) < uintptr(n) {
		panic("index out of bounds")
	}

	var res C.cublasStatus_t
	x.WithPtr(func(xPtr unsafe.Pointer) {
		y.WithPtr(func(yPtr unsafe.Pointer) {
			if h.PointerMode() == Host {
				pointerizeInputs(&alpha)
				res = C.cublasDaxpy(h.handle, safeIntToC(n), (*C.double)(alpha.(*float64)),
					(*C.double)(xPtr), safeIntToC(incx),
					(*C.double)(yPtr), safeIntToC(incy))
			} else {
				b := alpha.(cuda.Buffer)
				if b.Size() < 8 {
					panic("buffer underflow")
				}
				b.WithPtr(func(alphaPtr unsafe.Pointer) {
					res = C.cublasDaxpy(h.handle, safeIntToC(n), (*C.double)(alphaPtr),
						(*C.double)(xPtr), safeIntToC(incx),
						(*C.double)(yPtr), safeIntToC(incy))
				})
			}
		})
	})

	return newError("cublasDaxpy", res)
}

// Isamax gets the index of the first single-precision
// vector component with the max absolute value.
// The resulting indices start at one, not zero.
//
// The result argument's type depends on the pointer mode.
// In the Host pointer mode, use *int.
// In the Device pointer mode, use cuda.Buffer.
//
// This must be called inside the cuda.Context.
func (h *Handle) Isamax(n int, x cuda.Buffer, incx int, result interface{}) error {
	if n < 0 {
		panic("size out of bounds")
	} else if stridedSize(x.Size()/4, incx) < uintptr(n) {
		panic("index out of bounds")
	}

	var res C.cublasStatus_t
	x.WithPtr(func(xPtr unsafe.Pointer) {
		if h.PointerMode() == Host {
			var resInt C.int
			res = C.cublasIsamax(h.handle, safeIntToC(n), (*C.float)(xPtr),
				safeIntToC(incx), &resInt)
			*(result.(*int)) = int(resInt)
		} else {
			b := result.(cuda.Buffer)
			if b.Size() < 4 {
				panic("buffer underflow")
			}
			b.WithPtr(func(resPtr unsafe.Pointer) {
				res = C.cublasIsamax(h.handle, safeIntToC(n), (*C.float)(xPtr),
					safeIntToC(incx), (*C.int)(resPtr))
			})
		}
	})

	return newError("cublasIsamax", res)
}

// Idamax is like Isamax, but for double-precision.
//
// The result argument's type depends on the pointer mode.
// In the Host pointer mode, use *int.
// In the Device pointer mode, use cuda.Buffer.
//
// This must be called inside the cuda.Context.
func (h *Handle) Idamax(n int, x cuda.Buffer, incx int, result interface{}) error {
	if n < 0 {
		panic("size out of bounds")
	} else if stridedSize(x.Size()/8, incx) < uintptr(n) {
		panic("index out of bounds")
	}

	var res C.cublasStatus_t
	x.WithPtr(func(xPtr unsafe.Pointer) {
		if h.PointerMode() == Host {
			var resInt C.int
			res = C.cublasIdamax(h.handle, safeIntToC(n), (*C.double)(xPtr),
				safeIntToC(incx), &resInt)
			*(result.(*int)) = int(resInt)
		} else {
			b := result.(cuda.Buffer)
			if b.Size() < 4 {
				panic("buffer underflow")
			}
			b.WithPtr(func(resPtr unsafe.Pointer) {
				res = C.cublasIdamax(h.handle, safeIntToC(n), (*C.double)(xPtr),
					safeIntToC(incx), (*C.int)(resPtr))
			})
		}
	})

	return newError("cublasIdamax", res)
}

// Sasum sums the absolute values of the components in a
// single-precision vector.
//
// The result argument's type depends on the pointer mode.
// In the Host pointer mode, use *float32.
// In the Device pointer mode, use cuda.Buffer.
//
// This must be called inside the cuda.Context.
func (h *Handle) Sasum(n int, x cuda.Buffer, incx int, result interface{}) error {
	f := func(arg1 C.cublasHandle_t, arg2 C.int, arg3 *C.float, arg4 C.int,
		arg5 *C.float) C.cublasStatus_t {
		return C.cublasSasum(arg1, arg2, arg3, arg4, arg5)
	}
	return newError("cublasSasum", h.norm32(n, x, incx, result, f))
}

// Dasum is like Sasum, but for double-precision.
//
// The result argument's type depends on the pointer mode.
// In the Host pointer mode, use *float64.
// In the Device pointer mode, use cuda.Buffer.
//
// This must be called inside the cuda.Context.
func (h *Handle) Dasum(n int, x cuda.Buffer, incx int, result interface{}) error {
	f := func(arg1 C.cublasHandle_t, arg2 C.int, arg3 *C.double, arg4 C.int,
		arg5 *C.double) C.cublasStatus_t {
		return C.cublasDasum(arg1, arg2, arg3, arg4, arg5)
	}
	return newError("cublasDasum", h.norm64(n, x, incx, result, f))
}

// Snrm2 computes the Euclidean norm of a single-precision
// vector.
//
// The result argument's type depends on the pointer mode.
// In the Host pointer mode, use *float32.
// In the Device pointer mode, use cuda.Buffer.
//
// This must be called inside the cuda.Context.
func (h *Handle) Snrm2(n int, x cuda.Buffer, incx int, result interface{}) error {
	f := func(arg1 C.cublasHandle_t, arg2 C.int, arg3 *C.float, arg4 C.int,
		arg5 *C.float) C.cublasStatus_t {
		return C.cublasSnrm2(arg1, arg2, arg3, arg4, arg5)
	}
	return newError("cublasSnrm2", h.norm32(n, x, incx, result, f))
}

// Dnrm2 is like Snrm2, but for double-precision.
//
// The result argument's type depends on the pointer mode.
// In the Host pointer mode, use *float64.
// In the Device pointer mode, use cuda.Buffer.
//
// This must be called inside the cuda.Context.
func (h *Handle) Dnrm2(n int, x cuda.Buffer, incx int, result interface{}) error {
	f := func(arg1 C.cublasHandle_t, arg2 C.int, arg3 *C.double, arg4 C.int,
		arg5 *C.double) C.cublasStatus_t {
		return C.cublasDnrm2(arg1, arg2, arg3, arg4, arg5)
	}
	return newError("cublasDnrm2", h.norm64(n, x, incx, result, f))
}

func (h *Handle) norm32(n int, x cuda.Buffer, incx int, result interface{},
	f func(C.cublasHandle_t, C.int, *C.float, C.int, *C.float) C.cublasStatus_t) C.cublasStatus_t {
	if n < 0 {
		panic("size out of bounds")
	} else if stridedSize(x.Size()/4, incx) < uintptr(n) {
		panic("index out of bounds")
	}

	var res C.cublasStatus_t
	x.WithPtr(func(xPtr unsafe.Pointer) {
		if h.PointerMode() == Host {
			res = f(h.handle, safeIntToC(n), (*C.float)(xPtr),
				safeIntToC(incx), (*C.float)(result.(*float32)))
		} else {
			b := result.(cuda.Buffer)
			if b.Size() < 4 {
				panic("buffer underflow")
			}
			b.WithPtr(func(resPtr unsafe.Pointer) {
				res = f(h.handle, safeIntToC(n), (*C.float)(xPtr),
					safeIntToC(incx), (*C.float)(resPtr))
			})
		}
	})

	return res
}

func (h *Handle) norm64(n int, x cuda.Buffer, incx int, result interface{},
	f func(C.cublasHandle_t, C.int, *C.double, C.int,
		*C.double) C.cublasStatus_t) C.cublasStatus_t {
	if n < 0 {
		panic("size out of bounds")
	} else if stridedSize(x.Size()/8, incx) < uintptr(n) {
		panic("index out of bounds")
	}

	var res C.cublasStatus_t
	x.WithPtr(func(xPtr unsafe.Pointer) {
		if h.PointerMode() == Host {
			res = f(h.handle, safeIntToC(n), (*C.double)(xPtr),
				safeIntToC(incx), (*C.double)(result.(*float64)))
		} else {
			b := result.(cuda.Buffer)
			if b.Size() < 8 {
				panic("buffer underflow")
			}
			b.WithPtr(func(resPtr unsafe.Pointer) {
				res = f(h.handle, safeIntToC(n), (*C.double)(xPtr),
					safeIntToC(incx), (*C.double)(resPtr))
			})
		}
	})

	return res
}

func stridedSize(totalCount uintptr, inc int) uintptr {
	if inc == 0 {
		panic("zero increment")
	} else if inc < 0 {
		inc = -inc
	}
	// Do this in such a way that we never risk overflow.
	res := totalCount / uintptr(inc)
	if totalCount%uintptr(inc) != 0 {
		res++
	}
	return res
}
