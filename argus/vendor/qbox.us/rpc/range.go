package rpc

import (
	"strconv"
	"strings"
)

func parseRange1(rg string, fsize int64) (from int64, to int64, ok bool) {

	pos := strings.Index(rg, "-")
	if pos == -1 {
		return
	}

	from1 := strings.Trim(rg[:pos], " \t")
	to1 := strings.Trim(rg[pos+1:], " \t")

	var err error
	if from1 != "" {
		from, err = strconv.ParseInt(from1, 10, 64)
		if err != nil {
			return
		}
		if to1 != "" { // start-end
			to, err = strconv.ParseInt(to1, 10, 64)
			if err != nil {
				return
			}
			if to >= fsize {
				to = fsize - 1
			}
			// start和end是闭区间[start, end]
			// from和to是右半开区间[from, to)
			to++
		} else { // val-
			to = fsize
		}
	} else { // -val
		if to1 == "" {
			return
		}
		from, err = strconv.ParseInt(to1, 10, 64)
		if err != nil {
			return
		}
		to = fsize
		from = to - from
		if from < 0 {
			from = 0
		}
	}

	ok = from < to && to <= fsize
	return
}

func ParseOneRange(rg2 string, fsize int64) (from int64, to int64, ok bool) {
	pos := strings.Index(rg2, "=")
	if pos == -1 {
		return
	}
	return parseRange1(rg2[pos+1:], fsize)
}

func ParseRange(rg2 string, fsize int64) (rgs [][2]int64, total int64, ok bool) {

	pos := strings.Index(rg2, "=")
	if pos == -1 {
		return
	}

	rgArray := strings.Split(rg2[pos+1:], ",")
	rgs = make([][2]int64, len(rgArray))

	var from, to int64
	for i, rg := range rgArray {
		from, to, ok = parseRange1(rg, fsize)
		if !ok {
			return
		}
		rgs[i] = [2]int64{from, to}
		total += to - from
	}
	return
}
