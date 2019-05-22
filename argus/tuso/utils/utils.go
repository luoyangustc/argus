package utils

import (
	"encoding/binary"
	"math"

	"github.com/qiniu/http/httputil.v1"
)

func Float32ToByte(value float32) []byte {
	bits := math.Float32bits(value)
	bytes := make([]byte, 4)
	binary.LittleEndian.PutUint32(bytes, bits)

	return bytes
}

func ByteToFloat32(bytes []byte) float32 {
	bits := binary.LittleEndian.Uint32(bytes)

	return math.Float32frombits(bits)
}

func WithHttpCode(err error, code int) error {
	return httputil.NewError(code, err.Error())
}
