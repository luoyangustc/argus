package model

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestNewMessageBody(t *testing.T) {

	var (
		id   string = "1234567890ab"
		body []byte = []byte("0987654321")
	)
	body = NewMessageBody(id, body)
	assert.Equal(t, 23, len(body))
	assert.Equal(t, 12, int(body[0]))
	assert.EqualValues(t, body[1:13], []byte(id))
	assert.EqualValues(t, body[13:], []byte("0987654321"))
}

func TestParseMessageBody(t *testing.T) {
	{
		body := []byte("001234567899876543210")
		body[0] = byte(10)
		id, body2, _ := ParseMessageBody(body)
		assert.Equal(t, "0123456789", id)
		assert.EqualValues(t, []byte("9876543210"), body2)
	}
}
