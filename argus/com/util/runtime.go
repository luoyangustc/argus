package util

import (
	"runtime/debug"

	"github.com/qiniu/log.v1"
)

// HandleCrash recovers panic, logs panic info and calls custom handlers.
func HandleCrash(handlers ...func(interface{})) {
	if r := recover(); r != nil {
		log.Errorf("Observed a panic: %#v \n%s", r, string(debug.Stack()))
		for _, fn := range handlers {
			fn(r)
		}
	}
}
