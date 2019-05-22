package teapot

import (
	"bytes"
	"fmt"
	"io/ioutil"
	"os"
	"reflect"
	"runtime"
	"strconv"
	"strings"
)

func ReplaceFilters(old, new *Teapot) {
	old.filters = new.filters
}

type StrTo string

// string to bool
func (f StrTo) Bool() (bool, error) {
	s := f.String()
	if s == "on" {
		return true, nil
	}
	return strconv.ParseBool(s)
}

// string to float32
func (f StrTo) Float32() (float32, error) {
	v, err := strconv.ParseFloat(f.String(), 32)
	if err != nil {
		return 0, err
	}
	return float32(v), nil
}

// string to float64
func (f StrTo) Float64() (float64, error) {
	v, err := strconv.ParseFloat(f.String(), 64)
	if err != nil {
		return 0, err
	}
	return v, nil
}

// string to int
func (f StrTo) Int() (int, error) {
	v, err := strconv.ParseInt(f.String(), 10, 32)
	if err != nil {
		return 0, err
	}
	return int(v), nil
}

// string to int8
func (f StrTo) Int8() (int8, error) {
	v, err := strconv.ParseInt(f.String(), 10, 8)
	if err != nil {
		return 0, err
	}
	return int8(v), nil
}

// string to int16
func (f StrTo) Int16() (int16, error) {
	v, err := strconv.ParseInt(f.String(), 10, 16)
	if err != nil {
		return 0, err
	}
	return int16(v), nil
}

// string to int32
func (f StrTo) Int32() (int32, error) {
	v, err := strconv.ParseInt(f.String(), 10, 32)
	if err != nil {
		return 0, err
	}
	return int32(v), nil
}

// string to int64
func (f StrTo) Int64() (int64, error) {
	v, err := strconv.ParseInt(f.String(), 10, 64)
	if err != nil {
		return 0, err
	}
	return int64(v), nil
}

// string to uint
func (f StrTo) Uint() (uint, error) {
	v, err := strconv.ParseUint(f.String(), 10, 32)
	if err != nil {
		return 0, err
	}
	return uint(v), nil
}

// string to uint8
func (f StrTo) Uint8() (uint8, error) {
	v, err := strconv.ParseUint(f.String(), 10, 8)
	if err != nil {
		return 0, err
	}
	return uint8(v), nil
}

// string to uint16
func (f StrTo) Uint16() (uint16, error) {
	v, err := strconv.ParseUint(f.String(), 10, 16)
	if err != nil {
		return 0, err
	}
	return uint16(v), nil
}

// string to uint31
func (f StrTo) Uint32() (uint32, error) {
	v, err := strconv.ParseUint(f.String(), 10, 32)
	if err != nil {
		return 0, err
	}
	return uint32(v), nil
}

// string to uint64
func (f StrTo) Uint64() (uint64, error) {
	v, err := strconv.ParseUint(f.String(), 10, 64)
	if err != nil {
		return 0, err
	}
	return uint64(v), nil
}

// string to slice string
func (f StrTo) Strings(delimiters ...string) []string {
	delimiter := ","
	if len(delimiters) > 0 {
		delimiter = delimiters[0]
	}
	parts := strings.Split(f.String(), delimiter)
	for i, v := range parts {
		parts[i] = strings.TrimSpace(v)
	}
	return parts
}

func (f StrTo) String() string {
	return string(f)
}

// convert any type to string
func ToStr(value interface{}) (s string) {
	val := indirectValue(reflect.ValueOf(value))
	switch val.Kind() {
	case reflect.Bool:
		s = strconv.FormatBool(val.Bool())
	case reflect.Float32:
		s = strconv.FormatFloat(val.Float(), 'f', -1, 32)
	case reflect.Float64:
		s = strconv.FormatFloat(val.Float(), 'f', -1, 64)
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64:
		s = strconv.FormatInt(val.Int(), 10)
	case reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64:
		s = strconv.FormatUint(val.Uint(), 10)
	case reflect.String:
		s = val.String()
	case reflect.Array, reflect.Slice:
		if val.Type().Elem().Kind() == reflect.Uint8 {
			s = string(val.Bytes())
		}
	}
	if s == "" {
		s = fmt.Sprint(value)
	}
	return s
}

// reflect indirect of reflect.Value
func indirectValue(val reflect.Value) reflect.Value {
	for val.Kind() == reflect.Ptr {
		val = val.Elem()
	}
	return val
}

// reflect indirect of reflect.Type
func indirectType(typ reflect.Type) reflect.Type {
	for typ.Kind() == reflect.Ptr {
		typ = typ.Elem()
	}
	return typ
}

func convertStringAsType(str string, typ reflect.Type) (value reflect.Value) {
	var v interface{}

	iTyp := indirectType(typ)

	switch iTyp.Kind() {
	case reflect.Bool:
		v, _ = StrTo(str).Bool()
	case reflect.Float32:
		v, _ = StrTo(str).Float32()
	case reflect.Float64:
		v, _ = StrTo(str).Float64()
	case reflect.Int:
		v, _ = StrTo(str).Int()
	case reflect.Int8:
		v, _ = StrTo(str).Int8()
	case reflect.Int16:
		v, _ = StrTo(str).Int16()
	case reflect.Int32:
		v, _ = StrTo(str).Int32()
	case reflect.Int64:
		v, _ = StrTo(str).Int64()
	case reflect.Uint:
		v, _ = StrTo(str).Uint()
	case reflect.Uint8:
		v, _ = StrTo(str).Uint8()
	case reflect.Uint16:
		v, _ = StrTo(str).Uint16()
	case reflect.Uint32:
		v, _ = StrTo(str).Uint32()
	case reflect.Uint64:
		v, _ = StrTo(str).Uint64()
	case reflect.String:
		v = str
	default:
		return
	}

	val := reflect.ValueOf(v)
	if val.Type().ConvertibleTo(iTyp) {
		value = val.Convert(iTyp)
	}
	return
}

var (
	dunno     = []byte("???")
	centerDot = []byte("·")
	dot       = []byte(".")
	slash     = []byte("/")
)

// stack returns a nicely formated stack frame, skipping skip frames
func Stack(skip int) []byte {
	buf := new(bytes.Buffer) // the returned data
	// As we loop, we open files and read them. These variables record the currently
	// loaded file.
	var lines [][]byte
	var lastFile string
	for i := skip; ; i++ { // Caller we care about is the user, 2 frames up
		pc, file, line, ok := runtime.Caller(i)
		if !ok {
			break
		}
		// Print this much at least.  If we can't find the source, it won't show.
		fmt.Fprintf(buf, "%s:%d (0x%x)\n", file, line, pc)
		if file != lastFile {
			data, err := ioutil.ReadFile(file)
			if err != nil {
				continue
			}
			lines = bytes.Split(data, []byte{'\n'})
			lastFile = file
		}
		fmt.Fprintf(buf, "\t%s: %s\n", function(pc), source(lines, line))
	}
	return buf.Bytes()
}

func FileSource(file string, line int) ([]byte, error) {
	data, err := ioutil.ReadFile(file)
	if err != nil {
		return nil, err
	}
	lines := bytes.Split(data, []byte{'\n'})
	return source(lines, line), nil
}

// source returns a space-trimmed slice of the n'th line.
func source(lines [][]byte, n int) []byte {
	n-- // in stack trace, lines are 1-indexed but our array is 0-indexed
	if n < 0 || n >= len(lines) {
		return dunno
	}
	return bytes.Trim(lines[n], " \t")
}

// function returns, if possible, the name of the function containing the PC.
func function(pc uintptr) []byte {
	fn := runtime.FuncForPC(pc)
	if fn == nil {
		return dunno
	}
	name := []byte(fn.Name())
	// The name includes the path name to the package, which is unnecessary
	// since the file name is already included.  Plus, it has center dots.
	// That is, we see
	//	runtime/debug.*T·ptrmethod
	// and want
	//	*T.ptrmethod
	// Also the package path might contains dot (e.g. code.google.com/...),
	// so first eliminate the path prefix
	if lastslash := bytes.LastIndex(name, slash); lastslash >= 0 {
		name = name[lastslash+1:]
	}
	if period := bytes.Index(name, dot); period >= 0 {
		name = name[period+1:]
	}
	name = bytes.Replace(name, centerDot, dot, -1)
	return name
}

type brush func(string) string

func newBrush(color string) brush {
	pre := "\033[1;"
	reset := "\033[0m"
	return func(text string) string {
		if color == "" {
			return text
		}
		return pre + color + "m" + text + reset
	}
}

func getEnv(key, def string) string {
	value := os.Getenv(key)
	if value == "" {
		value = def
	}
	return value
}
