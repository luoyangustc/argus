package config

import (
	"context"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestDumpSimpleStruct(t *testing.T) {

	{
		var a = struct {
			A string
			B StaticConfigValue
		}{
			A: "xxx",
			B: NewStaticConfigValue(func() interface{} { return 100 }),
		}

		m, err := DumpSimpleStruct(context.Background(), a)
		assert.NoError(t, err)
		assert.Equal(t, "xxx", m["A"])
		assert.Equal(t, 100, m["B"])
	}

}

func TestDumpJsonConfig(t *testing.T) {
	{
		var a = struct {
			A  string
			SK string
		}{
			A:  "goodfood",
			SK: "DoDz1cdD6AbXLzY6ss_dT5VWvdmDzg0KhSikgTqI",
		}
		r := DumpJsonConfig(a)
		assert.Contains(t, r, "goodfood")
		assert.Contains(t, r, "******")
		assert.NotContains(t, r, "DoDz1cdD6AbXLzY6ss_dT5VWvdmDzg0KhSikgTqI")
	}
}
