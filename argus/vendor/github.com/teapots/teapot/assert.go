// simple test utils
package teapot

import (
	"fmt"
	"reflect"
	"runtime"
	"testing"
)

type Assert struct {
	T *testing.T
}

func (t *Assert) NoError(err error) {
	if err != nil {
		t.T.Errorf("expected no error but get `%v`\n%s", err, callerInfo())
		t.T.FailNow()
	}
}

func (t *Assert) Nil(ni interface{}) {
	val := reflect.ValueOf(ni)
	kind := val.Kind()
	if ni != nil && !(kind >= reflect.Chan && kind <= reflect.Slice && val.IsNil()) {
		t.T.Errorf("expected nil\n%s", callerInfo())
		t.T.FailNow()
	}
}

func (t *Assert) NotNil(ni interface{}) {
	val := reflect.ValueOf(ni)
	kind := val.Kind()
	if ni == nil || kind >= reflect.Chan && kind <= reflect.Slice && val.IsNil() {
		t.T.Errorf("expected not nil\n%s", callerInfo())
		t.T.FailNow()
	}
}

func (t *Assert) True(b bool) {
	if !b {
		t.T.Errorf("expected true but get false\n%s", callerInfo())
		t.T.FailNow()
	}
}

func (t *Assert) False(b bool) {
	if b {
		t.T.Errorf("expected false but get true\n%s", callerInfo())
		t.T.FailNow()
	}
}

func callerInfo() string {
	file := ""
	line := 0
	ok := false

	for i := 0; i < 3; i++ {
		_, file, line, ok = runtime.Caller(i)
		if !ok {
			return ""
		}
	}

	source, _ := FileSource(file, line)
	return fmt.Sprintf(`%s:%d %s`, file, line, source)
}
