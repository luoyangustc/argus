package sigmoid

import (
	"fmt"
	"strconv"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestSigmoid(t *testing.T) {
	value, _ := strconv.ParseFloat(fmt.Sprintf("%.3f", Sigmoid(0.35, 0.35, 0.8, 10)), 64)
	assert.Equal(t, 0.8, value)
	value, _ = strconv.ParseFloat(fmt.Sprintf("%.3f", Sigmoid(0.4, 0.35, 0.8, 10)), 64)
	assert.Equal(t, 0.868, value)
	value, _ = strconv.ParseFloat(fmt.Sprintf("%.3f", Sigmoid(0.5, 0.35, 0.8, 10)), 64)
	assert.Equal(t, 0.947, value)
	value, _ = strconv.ParseFloat(fmt.Sprintf("%.4f", Sigmoid(0.9997, 0.35, 0.8, 10)), 64)
	assert.Equal(t, 0.9997, value)
	value, _ = strconv.ParseFloat(fmt.Sprintf("%.3f", Sigmoid(1.0, 0.35, 0.8, 10)), 64)
	assert.Equal(t, 1.0, value)

	value, _ = strconv.ParseFloat(fmt.Sprintf("%.3f", Sigmoid(0.35, 0.35, 0.8, -1)), 64)
	assert.Equal(t, 0.8, value)
	value, _ = strconv.ParseFloat(fmt.Sprintf("%.3f", Sigmoid(0.35, 0.35, 0, 10)), 64)
	assert.Equal(t, 0.0, value)
}
