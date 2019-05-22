package teapot

import (
	"reflect"

	"github.com/teapots/inject"
)

type FilterFunc func() inject.Provider

type filters []*filter

type includeFilter filters

type exemptFilter filters

func (h filters) append(values ...*filter) filters {
outFor:
	for _, f := range values {
		for _, v := range h {
			if v.value == f.value {
				continue outFor
			}
		}
		h = append(h, f)
	}
	return h
}

func (h filters) remove(values ...*filter) filters {
	newFilters := make(filters, 0, len(h))
	for _, v := range h {
		exempt := false
	innerFor:
		for _, f := range values {
			if v.value == f.value {
				exempt = true
				break innerFor
			}
		}
		if !exempt {
			newFilters = append(newFilters, v)
		}
	}
	return newFilters
}

func Filter(handlers ...interface{}) Handler {
	return includeFilter(makeFilters(handlers))
}

func Exempt(handlers ...interface{}) Handler {
	return exemptFilter(makeFilters(handlers))
}

func makeFilters(handlers []interface{}) filters {
	res := make(filters, 0, len(handlers))
	for _, filter := range handlers {
		val := reflect.ValueOf(filter)

		ff := makeFilter(filter)
		f := newFilter(val, ff)
		res = append(res, f)
	}
	return res
}

func makeFilter(filter interface{}) (f FilterFunc) {
	switch arg := filter.(type) {
	case FilterFunc:
		f = arg

	case func() inject.Provider:
		f = arg

	case inject.Provide:
		f = wrapFilter(arg)

	default:
		val := reflect.ValueOf(filter)
		if val.Kind() != reflect.Func {
			panic("filter type must one of teapot.FilterFunc, inject.Provide, funcion")
		}

		f = wrapFilter(val.Interface())
	}
	return
}

func wrapFilter(arg interface{}) FilterFunc {
	return func() inject.Provider { return arg }
}

type filter struct {
	fun   FilterFunc
	value reflect.Value
}

func newFilter(value reflect.Value, fun FilterFunc) *filter {
	return &filter{
		fun:   fun,
		value: value,
	}
}
