package config

import (
	"context"
	"errors"
)

var (
	ErrConfigNotExist = errors.New("config not exist")
)

//----------------------------------------------------------------------------//

type ConfigValue interface {
	Value(context.Context) (interface{}, error)
}

type StaticConfigValue interface {
	ConfigValue
	Find(func(interface{}) interface{}) StaticConfigValue
}

type WatchConfigValue interface {
	ConfigValue
	Register(func(interface{}) error)
	Find(func(interface{}) interface{}) WatchConfigValue
}

type ConfigValues interface {
	Values(context.Context) ([]interface{}, []interface{}, error)
}

type WatchPrefixConfigValues interface {
	ConfigValues
	Register(func(interface{}, interface{}) error)
}
