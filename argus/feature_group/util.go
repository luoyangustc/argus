package feature_group

import (
	"encoding/binary"
	"math"
)

func FormatFloat32(byteorder binary.ByteOrder, f float32, p []byte) {
	byteorder.PutUint32(p, math.Float32bits(f))
}

func FormatFloat32s(byteorder binary.ByteOrder, fs []float32, buf []byte) {
	for i, j := 0, 0; i < len(fs); i, j = i+1, j+4 {
		FormatFloat32(byteorder, fs[i], buf[j:j+4])
	}
}

func ParseFloat32(byteorder binary.ByteOrder, p []byte) float32 {
	return math.Float32frombits(byteorder.Uint32(p))
}

func ParseFloat32Buf(byteorder binary.ByteOrder, p []byte, buf []float32) {
	for i, j := 0, 0; i < len(p); i, j = i+4, j+1 {
		buf[j] = ParseFloat32(byteorder, p[i:i+4])
	}
}

func BigEndianToLittleEndian(a []byte) []byte {
	b := make([]byte, len(a))
	for i := 0; i < len(a); i += 4 {
		r := binary.BigEndian.Uint32(a[i : i+4])
		binary.LittleEndian.PutUint32(b[i:], r)
	}
	return b
}

func UniqueStringSlice(src []string) []string {
	keys := make(map[string]struct{})
	res := []string{}
	for _, entry := range src {
		if _, v := keys[entry]; !v {
			keys[entry] = struct{}{}
			res = append(res, entry)
		}
	}
	return res
}

func StringArrayContains(s []string, e string) bool {
	for _, a := range s {
		if a == e {
			return true
		}
	}
	return false
}
