package workers

import (
	"encoding/json"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestInferenceImage(t *testing.T) {

	format := newInferenceImageRequestFormat("")
	assert.Equal(t,
		`{"data":{"uri":"xxx"}}`,
		format.Format(InferenceImageTask{URI: "xxx"}).String())
	assert.Equal(t,
		`{"data":{"uri":"xxx"},"params":{"limit":1}}`,
		format.Format(InferenceImageTask{URI: "xxx", Params: json.RawMessage(`{"limit":1}`)}).String())
}
