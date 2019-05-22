package time

import (
	"time"
)

func Nanoseconds() int64 {
	return time.Now().UnixNano()
}

func Seconds() int64 {
	return time.Now().Unix()
}
