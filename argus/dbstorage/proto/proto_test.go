package proto

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func Test_GetValidBool(t *testing.T) {
	res, err := GetValidBool("true")
	assert.Nil(t, err)
	assert.Equal(t, true, res)

	res, err = GetValidBool("false")
	assert.Nil(t, err)
	assert.Equal(t, false, res)

	_, err = GetValidBool("a")
	assert.Equal(t, "not a bool", err.Error())
}

func Test_GetValidMode(t *testing.T) {
	res, err := GetValidMode("")
	assert.Nil(t, err)
	assert.Equal(t, MODE_SINGLE, res)

	res, err = GetValidMode(MODE_SINGLE)
	assert.Nil(t, err)
	assert.Equal(t, MODE_SINGLE, res)

	res, err = GetValidMode(MODE_LARGEST)
	assert.Nil(t, err)
	assert.Equal(t, MODE_LARGEST, res)

	_, err = GetValidMode("test")
	assert.Equal(t, "invalid mode", err.Error())
}

func Test_ErrorLogString(t *testing.T) {
	log := ErrorLog{
		Uri:     "uri",
		Code:    400,
		Message: "error",
	}
	assert.Equal(t, "uri : 400 : error", log.String())
}
