package util

import (
	"fmt"
	"math"
	"regexp"
	"strconv"
)

var rangePattern = regexp.MustCompile(`^(-?[0-9]{0,20})-(-?[0-9]{0,20})$`)

// Range represents an integer range "from-to".
type Range struct {
	from *int64
	to   *int64
	min  *int64
	max  *int64
}

func NewRange(from, to *int64) Range {
	var r Range
	if from != nil {
		v := *from
		r.from = &v
	}
	if to != nil {
		v := *to
		r.to = &v
	}
	return r
}

func Int64Ptr(i int64) *int64 {
	return &i
}

func ParseRange(r string) (ret Range, err error) {
	subs := rangePattern.FindStringSubmatch(r)
	if len(subs) < 3 {
		err = fmt.Errorf("invalid range format")
		return
	}
	if subs[1] != "" {
		from, e := strconv.ParseInt(subs[1], 10, 64)
		if e != nil {
			err = e
			return
		}
		ret.from = &from
	}
	if subs[2] != "" {
		to, e := strconv.ParseInt(subs[2], 10, 64)
		if e != nil {
			err = e
			return
		}
		ret.to = &to
	}
	return
}

func (t Range) Valid() bool {
	ret := true
	if t.min != nil && t.max != nil {
		ret = (*t.min <= *t.max) && ret
	}
	if t.from != nil && t.to != nil {
		ret = (*t.from <= *t.to) && ret
	}
	return ret
}

func (t Range) Count() int64 {
	if !t.Valid() {
		return 0
	}

	var from, to int64
	if v := t.realFrom(); v != nil {
		from = *v
	} else {
		from = math.MinInt64
	}
	if v := t.realTo(); v != nil {
		to = *v
	} else {
		to = math.MaxInt64
	}

	// NOTE: may overflow
	c := to - from + 1
	if c < 0 {
		return 0
	}
	return c
}

func (t Range) realFrom() *int64 {
	if t.from != nil && t.min != nil {
		if *t.from > *t.min {
			return t.from
		}
		return t.min
	} else if t.from != nil {
		return t.from
	} else if t.min != nil {
		return t.min
	}
	return nil
}

func (t Range) realTo() *int64 {
	if t.to != nil && t.max != nil {
		if *t.to < *t.max {
			return t.to
		}
		return t.max
	} else if t.to != nil {
		return t.to
	} else if t.max != nil {
		return t.max
	}
	return nil
}

func (t Range) String() string {
	var s string
	rf := t.realFrom()
	rt := t.realTo()
	if rf != nil && rt != nil && *rf > *rt {
		return ""
	}

	if rf != nil {
		s += strconv.FormatInt(*rf, 10)
	}
	s += "-"
	if rt != nil {
		s += strconv.FormatInt(*rt, 10)
	}
	return s
}

func (t *Range) SetLimit(min, max *int64) {
	if min != nil {
		m := *min
		t.min = &m
	}
	if max != nil {
		m := *max
		t.max = &m
	}
}
