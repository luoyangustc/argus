package assert

import (
	"encoding/binary"
	"math"
)

var FLOAT_ENDIAN binary.ByteOrder = binary.BigEndian

// var FLOAT_ENDIAN binary.ByteOrder = binary.LittleEndian

func BigEndian() {
	FLOAT_ENDIAN = binary.BigEndian
}

func LittleEndian() {
	FLOAT_ENDIAN = binary.LittleEndian
}

func FormatFloat32(f float32, p []byte) {
	FLOAT_ENDIAN.PutUint32(p, math.Float32bits(f))
}

func FormatFloat32s(fs []float32, buf []byte) {
	for i, j := 0, 0; i < len(fs); i, j = i+1, j+4 {
		FormatFloat32(fs[i], buf[j:j+4])
	}
}

func ParseFloat32(p []byte) float32 {
	return math.Float32frombits(FLOAT_ENDIAN.Uint32(p))
}

func ParseFloat32Buf(p []byte, buf []float32) []float32 {
	for i, j := 0, 0; i < len(p); i, j = i+4, j+1 {
		buf[j] = ParseFloat32(p[i : i+4])
	}
	return buf
}

func Float32ToByte(float float32) []byte {
	bits := math.Float32bits(float)
	bytes := make([]byte, 4)
	FLOAT_ENDIAN.PutUint32(bytes, bits)

	return bytes
}

func ByteToFloat32(bytes []byte) float32 {
	bits := FLOAT_ENDIAN.Uint32(bytes)

	return math.Float32frombits(bits)
}

func Float64ToByte(float float64) []byte {
	bits := math.Float64bits(float)
	bytes := make([]byte, 8)
	FLOAT_ENDIAN.PutUint64(bytes, bits)

	return bytes
}
