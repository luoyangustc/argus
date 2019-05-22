package broker

import (
	"errors"
	"fmt"
	"net/url"
	"reflect"
	"strconv"
)

func stringMarshal(v reflect.Value) []string {
	switch v.Type().Kind() {
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64:
		return []string{strconv.FormatInt(v.Int(), 10)}
	case reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64:
		return []string{strconv.FormatUint(v.Uint(), 10)}
	case reflect.String:
		return []string{v.String()}
	case reflect.Float32:
		return []string{fmt.Sprintf("%f", v.Float())}
	case reflect.Bool:
		return []string{fmt.Sprintf("%t", v.Bool())}
	case reflect.Ptr:
		return stringMarshal(v.Elem())
	case reflect.Slice:
		l := v.Len()
		r := make([]string, l)
		for j := 0; j < l; j++ {
			r[j] = stringMarshal(v.Index(j))[0]
		}
		return r
	default:
		panic("bad request type:" + v.Type().String())
	}
}

func stringUnMarshal(data []string, v reflect.Value) error {
	if len(data) == 0 {
		return nil
	}
	switch v.Type().Kind() {
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64:
		r, err := strconv.ParseInt(data[0], 10, 64)
		if err != nil {
			return err
		}
		v.SetInt(r)
	case reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64:
		r, err := strconv.ParseUint(data[0], 10, 64)
		if err != nil {
			return err
		}
		v.SetUint(r)
	case reflect.String:
		v.SetString(data[0])
	case reflect.Float32:
		r, err := strconv.ParseFloat(data[0], 32)
		if err != nil {
			return err
		}
		v.SetFloat(r)
	case reflect.Bool:
		r, err := strconv.ParseBool(data[0])
		if err != nil {
			return err
		}
		v.SetBool(r)
	case reflect.Ptr:
		stringUnMarshal(data, v.Elem())
	case reflect.Slice:
		l := len(data)
		s := reflect.MakeSlice(v.Type(), l, l)
		v.Set(s)

		for j := 0; j < l; j++ {
			v0 := s.Index(j)
			err := stringUnMarshal([]string{data[j]}, v0.Addr())
			if err != nil {
				return err
			}
		}
	default:
		panic("bad request type:" + v.Type().String())
	}
	return nil
}

// FormMarshal return the string as the form-urlencoding
// it cannot process the embbed struct
func FormMarshal(o interface{}) url.Values {
	vs := url.Values{}
	typ, v := reflect.TypeOf(o), reflect.ValueOf(o)
	if typ.Kind() == reflect.Ptr {
		v = v.Elem()
		typ = typ.Elem()
	}

	for i, fc := 0, typ.NumField(); i < fc; i++ {
		f, fv := typ.Field(i), v.Field(i)
		n := f.Tag.Get("json")
		if n == "-" {
			continue
		}
		vs[n] = stringMarshal(fv)
	}
	return vs
}

// FormUnMarshal parses the data and returns the data
func FormUnMarshal(data url.Values, o interface{}) error {
	typ, v := reflect.TypeOf(o), reflect.ValueOf(o)
	if typ.Kind() != reflect.Ptr {
		return errors.New("not ptr")
	}

	typ = typ.Elem()
	for i, fc := 0, typ.NumField(); i < fc; i++ {
		f := typ.Field(i)
		n := f.Tag.Get("json")
		if n == "-" {
			continue
		}

		vs := data[n]
		if len(vs) == 0 {
			continue
		}

		fv := v.Elem().Field(i)
		err := stringUnMarshal(vs, fv.Addr())
		if err != nil {
			return err
		}
	}
	return nil
}
