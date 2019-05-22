package vframe

var (
	modeRg     [2]int     = [2]int{0, 2}
	intervalRg [2]float64 = [2]float64{0, 10}
)

func CheckMode(mode int) (err error) {
	if mode < modeRg[0] || mode > modeRg[1] {
		return ErrInvalidMode
	}
	return nil
}

func CheckInterval(interval float64) (err error) {
	if (interval < intervalRg[0]) || (interval > intervalRg[1]) {
		return ErrInvalidInterval
	}
	return nil
}

func CheckStartTime(ss float64) (err error) {
	if ss < 0 {
		err = ErrInvalidStartTime
	}
	return
}

func CheckDuration(duration float64) (err error) {
	if duration < 0 {
		err = ErrInvalidDuration
	}
	return
}
