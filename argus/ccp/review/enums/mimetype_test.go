package enums

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestMimeTypeIsValid(t *testing.T) {
	assertion := assert.New(t)
	assertion.True(MimeTypeImage.IsValid())
	assertion.True(MimeTypeVideo.IsValid())
	assertion.True(MimeTypeLive.IsValid())
	assertion.False(MimeType("invalid").IsValid())
}
