package model

import "net/http"

type IHeaderValue interface {
	Merge(string, string, string) (bool, string)
}

type headerValueCopy struct{ key string }

func NewHeaderValueCopy(key string) IHeaderValue { return headerValueCopy{key: key} }
func (copy headerValueCopy) Merge(key, v1, v2 string) (bool, string) {
	if key != copy.key {
		return false, ""
	}
	return true, v2
}

type measure struct{ key string }

func NewMeasure(key string) IHeaderValue { return measure{key: key} }
func (m measure) Merge(key, v1, v2 string) (bool, string) {
	if key != m.key {
		return false, ""
	}
	if v1 == "" {
		return true, v2
	}
	return true, v1 + ";" + v2 // TODO 暂不做合并
}

type _HeaderValue struct {
	m func(string, string, string) (bool, string)
}

func (v _HeaderValue) Merge(key, v1, v2 string) (bool, string) { return v.m(key, v1, v2) }
func NewHeaderValueFunc(m func(key, v1, v2 string) (bool, string)) IHeaderValue {
	return _HeaderValue{m: m}
}

type IHeader interface {
	Merge(http.Header, http.Header)
}

type headerMerger struct {
	Values []IHeaderValue
}

func NewHeaderMerger(value ...IHeaderValue) IHeader {
	return headerMerger{Values: value}
}

func (hm headerMerger) Merge(h1, h2 http.Header) {
	for key := range h2 {
		for _, v0 := range hm.Values {
			if ok, v3 := v0.Merge(key, h1.Get(key), h2.Get(key)); ok {
				h1.Set(key, v3)
				break
			}
		}
	}
}

const XOriginA = "X-Origin-A"

var DefaultHeader = NewHeaderMerger(
	NewMeasure(XOriginA), // ?
)

func MergeHeader(h1, h2 http.Header) {
	DefaultHeader.Merge(h1, h2)
}
