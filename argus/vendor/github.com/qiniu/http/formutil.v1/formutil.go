package formutil

import (
	"reflect"
	"strings"
	"syscall"

	"net/http"
	"net/url"

	"github.com/qiniu/errors"
	"github.com/qiniu/http/misc/strconv"
)

// --------------------------------------------------------------------

func ParseForm(ret interface{}, req *http.Request, postOnly bool) (err error) {

	err = req.ParseForm()
	if err != nil {
		err = errors.Info(err, "formutil.ParseForm failed").Detail(err)
		return
	}

	var form url.Values
	if postOnly {
		form = req.PostForm
	} else {
		form = req.Form
	}

	return ParseValue(reflect.ValueOf(ret), form, "json")
}

// --------------------------------------------------------------------

func Parse(ret interface{}, form url.Values) (err error) {

	return ParseValue(reflect.ValueOf(ret), form, "json")
}

func ParseEx(ret interface{}, form url.Values, cate string) (err error) {

	return ParseValue(reflect.ValueOf(ret), form, cate)
}

func ParseValue(v reflect.Value, form url.Values, cate string) (err error) {

	if v.Kind() != reflect.Ptr {
		err = errors.Info(syscall.EINVAL, "formutil.ParseValue: ret.type != pointer")
		return
	}

	v = v.Elem()
	if v.Kind() != reflect.Struct {
		err = errors.Info(syscall.EINVAL, "formutil.ParseValue: ret.type != struct")
		return
	}

	t := v.Type()
	for i := 0; i < v.NumField(); i++ {
		sf := t.Field(i)
		if sf.Tag == "" { // no tag
			if sf.Anonymous {
				err = ParseValue(v.Field(i).Addr(), form, cate)
				if err != nil {
					return
				}
			}
			continue
		}
		jsonTag := sf.Tag.Get(cate)
		if jsonTag == "" { // no json tag, skip
			continue
		}
		tag, opts, err2 := parseTag(jsonTag)
		if err2 != nil {
			err = errors.Info(err2, "Parse struct field:", sf.Name).Detail(err2)
			return
		}
		sfv := v.Field(i)
		fv, ok := form[tag]
		if opts.fhas {
			if err = setHas(v, sf.Name, ok); err != nil {
				return
			}
		}
		if !ok {
			if !opts.fdefault { // 允许外部设置默认值
				sfv.Set(reflect.Zero(sf.Type))
			}
			continue
		} else if len(fv) == 0 {
			sfv.Set(reflect.Zero(sf.Type))
			continue
		}
		switch sfv.Kind() {
		case reflect.Slice:
			sft := sfv.Type()
			n := len(fv)
			slice := reflect.MakeSlice(sft, n, n)
			for i := 0; i < n; i++ {
				err = strconv.ParseValue(slice.Index(i), fv[i])
				if err != nil {
					err = errors.Info(err, "formutil.ParseValue: parse slice field -", sf.Name, i).Detail(err)
					return
				}
			}
			sfv.Set(slice)
		default:
			err = strconv.ParseValue(sfv, fv[0])
			if err != nil {
				err = errors.Info(err, "formutil.ParseValue: parse struct field -", sf.Name).Detail(err)
				return
			}
		}
	}
	return
}

// --------------------------------------------------------------------

func setHas(v reflect.Value, name string, has bool) (err error) {

	sfHas := v.FieldByName("Has" + name)
	if sfHas.Kind() != reflect.Bool {
		err = errors.New("Struct filed `Has" + name + "` not found or not bool")
		return
	}
	sfHas.SetBool(has)
	return
}

type tagParseOpts struct {
	fhas     bool
	fdefault bool
}

func parseTag(tag1 string) (tag string, opts tagParseOpts, err error) {

	if tag1 == "" {
		err = errors.New("Struct field has no tag")
		return
	}

	parts := strings.Split(tag1, ",")
	tag = parts[0]
	for i := 1; i < len(parts); i++ {
		switch parts[i] {
		case "has":
			opts.fhas = true
		case "default":
			opts.fdefault = true
		case "omitempty":
		default:
			err = errors.New("Unknown tag option: " + parts[i])
			return
		}
	}
	return
}

// --------------------------------------------------------------------
