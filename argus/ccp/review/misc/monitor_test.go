package misc

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestMonitor(t *testing.T) {
	assertion := assert.New(t)
	assertion.NotNil(ResponseTime("api", 0))
	assertion.NotNil(RequestsCounter("api", 0))
}
