package config

import (
	"context"
	"encoding/json"
	"fmt"
	"reflect"
	"regexp"
)

func WatchValueAsStaticValue(v WatchConfigValue) StaticConfigValue {
	return staticChildConfigValue{
		ConfigValue: v,
		find:        func(v interface{}) interface{} { return v },
	}
}

//----------------------------------------------------------------------------//

func GetInt64(ctx context.Context, value ConfigValue, defaultValue int64) int64 {
	v1, err := value.Value(ctx)
	if err != nil || v1 == nil {
		return defaultValue
	}
	v2, ok := v1.(int64)
	if !ok {
		return defaultValue
	}
	return v2
}

func GetInt(ctx context.Context, value ConfigValue, defaultValue int) int {
	v1, err := value.Value(ctx)
	if err != nil || v1 == nil {
		return defaultValue
	}
	v2, ok := v1.(int)
	if !ok {
		return defaultValue
	}
	return v2
}

func GetString(ctx context.Context, value ConfigValue, defaultValue string) string {
	v1, err := value.Value(ctx)
	if err != nil || v1 == nil {
		return defaultValue
	}
	v2, ok := v1.(string)
	if !ok {
		return defaultValue
	}
	return v2
}

////////////////////////////////////////////////////////////////////////////////

var _TypeConfigValue = reflect.TypeOf(new(ConfigValue)).Elem()
var _TypeConfigValues = reflect.TypeOf(new(ConfigValues)).Elem()

// DumpSimpleStruct ...
func DumpSimpleStruct(ctx context.Context, src interface{}) (map[string]interface{}, error) {
	var (
		m     = make(map[string]interface{})
		value = reflect.ValueOf(src)
		typ   = value.Type()
	)
	for i, n := 0, value.NumField(); i < n; i++ {
		field := value.Field(i)
		switch {
		case field.Type().Implements(_TypeConfigValue):
			rests := field.MethodByName("Value").Call([]reflect.Value{reflect.ValueOf(ctx)})
			if !rests[1].IsNil() {
				return nil, rests[1].Interface().(error)
			}
			m[typ.Field(i).Name] = rests[0].Interface()
		case field.Type().Implements(_TypeConfigValues):
			rests := field.MethodByName("Values").Call([]reflect.Value{reflect.ValueOf(ctx)})
			if !rests[2].IsNil() {
				return nil, rests[2].Interface().(error)
			}
			m := make(map[string]interface{})
			for i, k := range rests[0].Interface().([]interface{}) {
				v := rests[1].Interface().([]interface{})[i]
				m[fmt.Sprintf("%v", k)] = v
			}
			m[typ.Field(i).Name] = m
		default:
			m[typ.Field(i).Name] = field.Interface()
		}
	}
	return m, nil
}

func DumpJsonConfig(conf interface{}) string {
	buf, _ := json.Marshal(conf)
	return string(regexp.MustCompile(`"[a-zA-Z0-9_]{40}"`).ReplaceAll(buf, []byte(`"******"`)))
}
