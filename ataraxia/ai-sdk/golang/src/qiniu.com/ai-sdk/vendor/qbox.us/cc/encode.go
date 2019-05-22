package cc

import "encoding/base64"

func UrlsafeBase64Encode(val []byte) []byte {
	b := make([]byte, base64.URLEncoding.EncodedLen(len(val)))
	base64.URLEncoding.Encode(b, val)
	return b
}

func UrlsafeBase64Str(val []byte) string {
	b := make([]byte, base64.URLEncoding.EncodedLen(len(val)))
	base64.URLEncoding.Encode(b, val)
	return string(b)
}

func UrlsafeBase64Decode(val []byte) ([]byte, error) {
	b := make([]byte, base64.URLEncoding.DecodedLen(len(val)))
	n, err := base64.URLEncoding.Decode(b, val)
	return b[:n], err
}

func UrlsafeBase64DecodeStr(val string) ([]byte, error) {
	b := make([]byte, base64.URLEncoding.DecodedLen(len(val)))
	n, err := base64.URLEncoding.Decode(b, []byte(val))
	return b[:n], err
}

func Base64Encode(val []byte) []byte {
	b := make([]byte, base64.StdEncoding.EncodedLen(len(val)))
	base64.StdEncoding.Encode(b, val)
	return b
}

func Base64Decode(val []byte) ([]byte, error) {
	b := make([]byte, base64.StdEncoding.DecodedLen(len(val)))
	n, err := base64.StdEncoding.Decode(b, val)
	return b[:n], err
}
