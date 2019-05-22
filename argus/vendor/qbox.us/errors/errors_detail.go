package errors

import (
	"fmt"

	"github.com/qiniu/log.v1"
)

const prefix = " ==> "

// --------------------------------------------------------------------

type errorDetailer interface {
	ErrorDetail() string
}

func Detail(err error) string {
	if e, ok := err.(errorDetailer); ok {
		return e.ErrorDetail()
	}
	return prefix + err.Error()
}

// --------------------------------------------------------------------

type ErrorInfo struct {
	Err error
	Why error
	Cmd []interface{}
}

func Info(err error, cmd ...interface{}) *ErrorInfo {
	return &ErrorInfo{Cmd: cmd, Err: Err(err)}
}

func (r *ErrorInfo) Cause() error {
	return r.Err
}

func (r *ErrorInfo) Error() string {
	return r.Err.Error()
}

func (r *ErrorInfo) ErrorDetail() string {
	e := prefix + r.Err.Error() + " ~ " + fmt.Sprint(r.Cmd...)
	if r.Why != nil {
		e += "\n" + Detail(r.Why)
	}
	return e
}

func (r *ErrorInfo) Detail(err error) *ErrorInfo {
	r.Why = err
	return r
}

func (r *ErrorInfo) Method() (cmd string, ok bool) {
	if len(r.Cmd) > 0 {
		cmd, ok = r.Cmd[0].(string)
	}
	return
}

func (r *ErrorInfo) LogMessage() string {
	detail := r.ErrorDetail()
	if cmd, ok := r.Method(); ok {
		detail = cmd + " failed:\n" + detail
	}
	return detail
}

func (r *ErrorInfo) Warn() *ErrorInfo {
	log.Std.Output("", log.Lwarn, 2, r.LogMessage())
	return r
}

// --------------------------------------------------------------------

type causer interface {
	Cause() error
}

func Err(err error) error {
	if e, ok := err.(causer); ok {
		if diag := e.Cause(); diag != nil {
			return diag
		}
	}
	return err
}

// --------------------------------------------------------------------
