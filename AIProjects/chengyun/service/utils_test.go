package service

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestCameraFilters(t *testing.T) {
	a := assert.New(t)
	id, licenceID := "NHPZXPZ00000001320180410031641380", "沪DQ0978"
	a.True(CameraMatchFilters(id, licenceID))

	id, licenceID = "NHPZXPZ00000001120180410031641380", "沪DQ0978"
	a.True(CameraMatchFilters(id, licenceID))

	id, licenceID = "NHPZXPZ00000001220180410031641380", "沪DQ0978"
	a.True(CameraMatchFilters(id, licenceID))

	id, licenceID = "NHPZXPZ00000001320180410031641380", "浙DQ0978"
	a.False(CameraMatchFilters(id, licenceID))

	id, licenceID = "", "浙DQ0978"
	a.False(CameraMatchFilters(id, licenceID))
}
