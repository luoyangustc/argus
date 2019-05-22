package config

import (
	"context"
	"sync"
)

type multiStaticConfigValue struct {
	values []ConfigValue
}

func NewMultiStaticConfigValue(value ...ConfigValue) StaticConfigValue {
	return multiStaticConfigValue{values: value}
}

func (v multiStaticConfigValue) Value(ctx context.Context) (vv interface{}, err error) {
	for _, value := range v.values {
		if vv, err = value.Value(ctx); err == nil && vv != nil {
			return
		}
	}
	if err == nil {
		err = ErrConfigNotExist
	}
	return
}

func (v multiStaticConfigValue) Find(find func(interface{}) interface{}) StaticConfigValue {
	return staticChildConfigValue{
		ConfigValue: v,
		find:        find,
	}
}

//----------------------------------------------------------------------------//

type multiWatchConfigValue struct {
	values []WatchConfigValue
}

func NewMultiWatchConfigValue(value ...WatchConfigValue) WatchConfigValue {
	return &multiWatchConfigValue{values: value}
}

func (v *multiWatchConfigValue) Value(ctx context.Context) (vv interface{}, err error) {
	for _, value := range v.values {
		if vv, err = value.Value(ctx); err == nil && vv != nil {
			return
		}
	}
	if err == nil {
		err = ErrConfigNotExist
	}
	return
}

func (v *multiWatchConfigValue) Register(set func(interface{}) error) {
	var (
		last = -1
		mtx  = new(sync.Mutex)

		set2 = func(index int) func(interface{}) error {
			return func(vv interface{}) error {
				mtx.Lock()
				defer mtx.Unlock()

				if last >= 0 {
					if index > last {
						return nil
					}
					if vv != nil {
						if index <= last {
							last = index
							return set(vv)
						}
					} else {
						if index != last {
							return nil
						}
					}
				}

				for j := last + 1; j < len(v.values); j++ {
					if j == index {
						last = j
						return set(vv)
					}
					if vv2, err := v.values[j].Value(context.Background()); err == nil {
						last = j
						return set(vv2)
					}
				}

				return set(nil)
			}
		}
	)
	for i, value := range v.values {
		value.Register(set2(i))
	}
}

func (v *multiWatchConfigValue) Find(find func(interface{}) interface{}) WatchConfigValue {
	return &watchChildConfigValue{
		WatchConfigValue: v,
		find:             find,
		values:           make([]interface{}, 0),
	}
}
