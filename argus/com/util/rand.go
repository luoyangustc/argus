package util

import (
	crand "crypto/rand"
	"encoding/binary"
	"errors"
	"math"
	mrand "math/rand"
	"time"
)

const alphabet = "0123456789" +
	"abcdefghijklmnopqrstuvwxyz" +
	"ABCDEFGHIJKLMNOPQRSTUVWXYZ" +
	`!@#$%^&*()_-+=[]{}\|,./<>?~:;`

const RandomAlphabetLength = len(alphabet)

func init() {
	seed, err := Int64()
	if err != nil {
		// todo: warning
		mrand.Seed(time.Now().UnixNano())
	} else {
		mrand.Seed(seed)
	}
}

// MustGenRandomString generates a random string with given length and alphabet size
// length is positive, and base is in [1, 91], or it will panic
// it does not return an error, and is given a long name for use with caution.
func MustGenRandomString(length, base int) string {
	s, e := RandomString(length, base)
	if e != nil {
		panic(e.Error())
	}
	return s
}

func Int64() (int64, error) {
	var val int64
	if err := binary.Read(crand.Reader, binary.BigEndian, &val); err != nil {
		return 0, err
	}
	return val, nil
}

// RandomString generates a random string with given length and alphabet base
// base should be in the range of [1, 91]
func RandomString(length, base int) (string, error) {
	if base <= 0 || base > RandomAlphabetLength {
		return "", errors.New("invalid alphabet base size for generating random string")
	}
	dict := alphabet[:base]

	section := bits(uint64(base))
	var mask uint64 = 1<<section - 1
	maxSection := int(64 / section)
	nuance := make([]byte, length)
	var u64 uint64
	for i, s := 0, 0; i < length; {
		if s < 1 {
			u64 = mrand.Uint64()
			s = maxSection
		}
		num := int(u64 & mask)
		u64 >>= section
		s--
		if num < base {
			nuance[i] = dict[num]
			i++
		}
	}

	return string(nuance), nil
}

func bits(u uint64) uint64 {
	if u == 0 {
		return 0
	}
	b := uint64(math.Ilogb(float64(u)))
	if u > 1<<b-1 {
		b++
	}
	return b
}
