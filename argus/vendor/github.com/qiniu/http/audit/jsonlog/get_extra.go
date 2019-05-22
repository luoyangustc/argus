package jsonlog

import (
	"net/http"
	"reflect"
)

// ----------------------------------------------------------

type extraWriter interface {
	ExtraWrite(key string, val interface{})
}

func getExtraWriter(w http.ResponseWriter) (ew extraWriter, ok bool) {

	v := reflect.ValueOf(w)
	v = reflect.Indirect(v)
	for v.Kind() == reflect.Struct {
		if fv := v.FieldByName("ResponseWriter"); fv.IsValid() {
			if ew, ok = fv.Interface().(extraWriter); ok {
				return
			}
			if fv.Kind() == reflect.Interface {
				fv = fv.Elem()
			}
			v = reflect.Indirect(fv)
		} else {
			break
		}
	}
	return
}

// ----------------------------------------------------------

type extraInt64Adder interface {
	ExtraAddInt64(key string, val int64)
}

func getExtraInt64Adder(w http.ResponseWriter) (ew extraInt64Adder, ok bool) {

	v := reflect.ValueOf(w)
	v = reflect.Indirect(v)
	for v.Kind() == reflect.Struct {
		if fv := v.FieldByName("ResponseWriter"); fv.IsValid() {
			if ew, ok = fv.Interface().(extraInt64Adder); ok {
				return
			}
			if fv.Kind() == reflect.Interface {
				fv = fv.Elem()
			}
			v = reflect.Indirect(fv)
		} else {
			break
		}
	}
	return
}

// ----------------------------------------------------------

type extraStringAdder interface {
	ExtraAddString(key string, val string)
}

func getExtraStringAdder(w http.ResponseWriter) (ew extraStringAdder, ok bool) {

	v := reflect.ValueOf(w)
	v = reflect.Indirect(v)
	for v.Kind() == reflect.Struct {
		if fv := v.FieldByName("ResponseWriter"); fv.IsValid() {
			if ew, ok = fv.Interface().(extraStringAdder); ok {
				return
			}
			if fv.Kind() == reflect.Interface {
				fv = fv.Elem()
			}
			v = reflect.Indirect(fv)
		} else {
			break
		}
	}
	return
}

// ----------------------------------------------------------

type extraBodyLogDisabler interface {
	ExtraDisableBodyLog()
}

func getExtraBodyLogDisabler(w http.ResponseWriter) (ew extraBodyLogDisabler, ok bool) {

	v := reflect.ValueOf(w)
	v = reflect.Indirect(v)
	for v.Kind() == reflect.Struct {
		if fv := v.FieldByName("ResponseWriter"); fv.IsValid() {
			if ew, ok = fv.Interface().(extraBodyLogDisabler); ok {
				return
			}
			if fv.Kind() == reflect.Interface {
				fv = fv.Elem()
			}
			v = reflect.Indirect(fv)
		} else {
			break
		}
	}
	return
}

// ----------------------------------------------------------
