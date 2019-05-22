package inject

import (
	"fmt"
	"reflect"
)

const (
	INJECT_MAX_RECURSIVE_LEVEL = 3
)

type typeError error

type Injector interface {
	TypeProvider

	Find(interface{}, string) error
	Invoke(interface{}) ([]reflect.Value, error)
	Apply(interface{}) error

	SetParent(Injector) Injector

	// private use
	provInvoker
}

type Object struct {
	Value interface{}
	Type  interface{}
	Name  string
}

type TypeProvider interface {
	Provide(provs ...interface{}) TypeProvider
	ProvideAs(prov interface{}, typ interface{}) TypeProvider
}

type provInvoker interface {
	get(string) *providerInfo
}

type invokeStatus map[string]bool

func (m invokeStatus) has(name string) bool {
	_, ok := m[name]
	return ok
}

func (m invokeStatus) set(name string) {
	m[name] = false
}

type invokeCache map[string][]reflect.Value

type injector struct {
	// use for store provider
	values map[string]*providerInfo

	// user for cache provider instance for current inject
	caches invokeCache

	parent Injector
}

func New() Injector {
	return &injector{
		values: make(map[string]*providerInfo),
		caches: make(invokeCache),
	}
}

func (inj *injector) Provide(provs ...interface{}) TypeProvider {
	for _, prov := range provs {
		inj.ProvideAs(prov, nil)
	}
	return inj
}

func (inj *injector) ProvideAs(prov interface{}, typ interface{}) TypeProvider {
	var obj Object
	switch p := prov.(type) {
	case Object:
		obj = p
	case *Object:
		obj = *p
	default:
		obj = Object{Value: prov}
	}

	if typ != nil {
		obj.Type = typ
	}

	info := newProvider(obj)

	// remove exists cache of provider
	delete(inj.caches, info.name)

	// replace with new prvoder info
	inj.values[info.name] = info
	return inj
}

func (inj *injector) Find(ptr interface{}, name string) error {
	val := reflect.ValueOf(ptr)
	if val.Kind() != reflect.Ptr {
		panic("need ptr instance")
	}
	provName := createName(indirectType(val.Type()), name)

	prov := inj.get(provName)
	if prov == nil {
		return fmt.Errorf("provider not found of type `%s`", provName)
	}

	status := make(invokeStatus)
	out, err := prov.invoke(inj, status)
	if err != nil {
		return err
	}

	if len(out) > 0 {
		ot := out[0]

		if !ot.IsValid() {
			return fmt.Errorf("provider value not valid of type `%s`", provName)
		}

		return assignValue(ot, val.Elem())
	}

	return nil
}

func (inj *injector) Invoke(prov interface{}) ([]reflect.Value, error) {
	status := make(invokeStatus)

	info := newProvider(Object{Value: prov})
	out, err := info.invoke(inj, status)

	if err != nil {
		return nil, fmt.Errorf("provider invoke err: %v", err)
	}

	// remove exists cache of provider
	delete(inj.caches, info.name)
	return out, nil
}

func (inj *injector) Apply(ptrStruct interface{}) error {
	// status use for check cycle dependencies in current apply flow
	status := make(invokeStatus)

	level := 0

	return inj.apply(ptrStruct, status, level)
}

func (inj *injector) apply(ptrStruct interface{}, status invokeStatus, level int) error {
	level += 1

	val := reflect.ValueOf(ptrStruct)
	elm := reflect.Indirect(val)

	if elm.Kind() != reflect.Struct {
		return typeError(fmt.Errorf("expected a <*struct> of %v", val))
	}

	typ := elm.Type()

	for i := 0; i < elm.NumField(); i++ {
		field := elm.Field(i)
		structField := typ.Field(i)

		if !field.CanSet() {
			continue
		}

		tagVal := structField.Tag.Get("inject")
		if tagVal == "-" {
			continue
		}

		if structField.Tag == "inject" || tagVal != "" {
			// create name of inject value
			provName := createName(indirectType(field.Type()), tagVal)

			prov := inj.get(provName)
			if prov == nil {
				return fmt.Errorf("provider not found for type %s:%v", provName, field)
			}

			out, err := prov.invoke(inj, status)

			if err != nil {
				return fmt.Errorf("provider invoke of type %s:%v err: %v", provName, field, err)
			}

			if len(out) > 0 {
				if !out[0].IsValid() {
					return fmt.Errorf("value not found for type %s:%v", provName, field)
				}

				assignValue(out[0], field)
			}

			continue
		}

		if level >= INJECT_MAX_RECURSIVE_LEVEL {
			continue
		}

		if field.CanInterface() {
			if field.Kind() == reflect.Struct {
				// restore to pointer struct
				field = field.Addr()
			}

			// child typeError should skip
			if err, ok := inj.apply(field.Interface(), status, level).(typeError); !ok && err != nil {
				return err
			}
		}
	}

	return nil
}

func (inj *injector) get(name string) *providerInfo {
	// get provider in current injector
	if prov := inj.values[name]; prov != nil {
		return prov
	}

	// back to parent injector
	if inj.parent != nil {
		return inj.parent.get(name)
	}

	return nil
}

// set parent injector
func (inj *injector) SetParent(parent Injector) Injector {
	inj.parent = parent
	return inj
}

func assignValue(out, elm reflect.Value) (err error) {
	if out.Type().AssignableTo(elm.Type()) {
		elm.Set(out)
	} else if out.Kind() == reflect.Ptr && out.Type().Elem().AssignableTo(elm.Type()) {
		elm.Set(out.Elem())
	} else {
		err = fmt.Errorf("unsupport value assignable from `%v` to `%v`", out.Type(), elm.Type())
	}
	return
}
