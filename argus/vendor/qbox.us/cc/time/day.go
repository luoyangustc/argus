package time

import (
	"time"
)

func SameDay(a, b time.Time) bool {
	return a.Day() == b.Day() && a.Month() == b.Month() && a.Year() == b.Year()
}

//00:00 of the day
func DayStart(t time.Time) time.Time {
	return time.Date(t.Year(), t.Month(), t.Day(), 0, 0, 0, 0, t.Location())
}

// check < 00:00 of the day
func BeforeTheDay(prev, cur time.Time) bool {
	midnight := DayStart(cur)
	return prev.Before(midnight)
}
