package server

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func Test_server_readAppStatus(t *testing.T) {
	s := server{mock: true, cfg: Config{StsHost: "http://10.200.30.13:5555"}}
	s.readAppStatus()
	assert.Equal(t, true, s.appExists("ava-terror-detect"))
	assert.Equal(t, false, s.appExists("xxx"))
	s.appStatusPage()
}
