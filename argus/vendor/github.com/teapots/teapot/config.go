package teapot

import (
	"os"
	"path/filepath"
	"reflect"
)

const (
	ModeDev  Mode = "dev"
	ModeTest      = "test"
	ModeProd      = "prod"
)

type Mode string

func (m Mode) IsDev() bool {
	return m == ModeDev
}

func (m Mode) IsTest() bool {
	return m == ModeTest
}

func (m Mode) IsProd() bool {
	return m == ModeProd
}

// any config system
type Configer interface {
	// return raw config value
	Find(name string) string
}

type Config struct {
	RunPath  string
	RunMode  Mode
	HttpAddr string
	HttpPort string

	Configer
}

func newConfig() *Config {
	root, _ := os.Getwd()
	root, _ = filepath.Abs(root)

	path := getEnv("RUN_PATH", root)
	mode := Mode(getEnv("RUN_MODE", string(ModeProd)))
	addr := getEnv("HTTP_ADDR", "127.0.0.1")
	port := getEnv("HTTP_PORT", "3000")

	return &Config{
		RunPath:  path,
		RunMode:  mode,
		HttpAddr: addr,
		HttpPort: port,
	}
}

func (c *Config) Find(name string, defaultValue ...string) string {
	if c.Configer != nil {
		value := c.Configer.Find(name)
		if value == "" && len(defaultValue) > 0 {
			return defaultValue[0]
		}
		return value
	}
	return ""
}

func (c *Config) Bind(ptr interface{}, name string) {
	if c.Configer != nil {
		val := reflect.ValueOf(ptr)
		if val.Kind() != reflect.Ptr {
			panic("Bind need pointer of type")
		}

		value := c.Find(name)
		if value == "" {
			return
		}

		elm := val.Elem()
		val = convertStringAsType(value, elm.Type())
		if elm.CanSet() && val.IsValid() {
			if elm.Kind() == reflect.Ptr {
				elm.Set(reflect.New(elm.Type().Elem()))
				elm = elm.Elem()
			}
			elm.Set(val)
		}
	}
}

func (c *Config) setParent(parent Configer) {
	c.Configer = parent
}
