package util

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func Test_Substring(t *testing.T) {
	var origin, res string
	origin = "ab中文cd_1234"
	res = Substring(origin, -1, 1)
	assert.Equal(t, res, "")

	res = Substring(origin, 0, 100)
	assert.Equal(t, res, "")

	res = Substring(origin, 3, 2)
	assert.Equal(t, res, "")

	res = Substring(origin, 1, 3)
	assert.Equal(t, res, "b中")
}

func Test_GetTagAndDesc(t *testing.T) {
	var origin, tag, desc string
	origin = "ab中文cd_12中文34.jpg"
	tag, desc = GetTagAndDesc(origin)
	assert.Equal(t, tag, "ab中文cd")
	assert.Equal(t, desc, "12中文34")

	origin = "ab中文cd_12中文34_abcd.jpg"
	tag, desc = GetTagAndDesc(origin)
	assert.Equal(t, tag, "ab中文cd")
	assert.Equal(t, desc, "12中文34_abcd")

	origin = "ab中文cd_12中文34_without_extension"
	tag, desc = GetTagAndDesc(origin)
	assert.Equal(t, tag, "ab中文cd")
	assert.Equal(t, desc, "12中文34_without_extension")
}

func Test_GetSha1(t *testing.T) {
	var origin []byte
	var res string
	origin = []byte("ab中文cd")
	res = GetSha1(origin)
	assert.Equal(t, res, "8c04f5c4f21738b591b0915c8cafca4317175dc1")
}

func Test_ArrayContains(t *testing.T) {
	array := []int{1, 2, 3}
	exist := ArrayContains(array, 2)
	assert.Equal(t, true, exist)

	exist = ArrayContains(array, 4)
	assert.Equal(t, false, exist)
}
