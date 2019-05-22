package model

import (
	"net/http"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestHeaderAdder(t *testing.T) {

	adder := NewHeaderMerger(
		NewHeaderValueCopy("A"),
		NewHeaderValueCopy("B"),
		NewHeaderValueCopy("C"),
		NewMeasure("D"),
		NewMeasure("E"),
	)

	var (
		h1 = http.Header(
			map[string][]string{
				"A": []string{"1"},
				"B": []string{"2"},
				"D": []string{"X,1"},
				"F": []string{"3"},
			},
		)
		h2 = http.Header(
			map[string][]string{
				"A": []string{"12"},
				"C": []string{"34"},
				"D": []string{"Y,1"},
				"E": []string{"Y,1"},
			},
		)
	)

	adder.Merge(h1, h2)

	assert.Equal(t, "12", h1.Get("A"))
	assert.Equal(t, "2", h1.Get("B"))
	assert.Equal(t, "34", h1.Get("C"))
	assert.Equal(t, "X,1;Y,1", h1.Get("D"))
	assert.Equal(t, "Y,1", h1.Get("E"))
	assert.Equal(t, "3", h1.Get("F"))

}
