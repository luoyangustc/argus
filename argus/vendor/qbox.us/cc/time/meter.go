package time

import (
	"time"
)

type Meter time.Time

func NewMeter() Meter {
	return Meter(time.Now())
}

func (tm Meter) Elapsed() int64 {
	return time.Now().Sub(time.Time(tm)).Nanoseconds()
}
