package feature_group

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

type mockFeatureAPI struct {
	Length int
}

func TestFeatureAPIs(t *testing.T) {
	var v1 = FeatureVersion("v1")
	var v2 = FeatureVersion("v2")
	var dv = v1
	var cv = v1
	var m = map[FeatureVersion]FeatureAPI{
		FeatureVersion("v1"): mockFeatureAPI{
			Length: 4,
		},
	}

	apis := NewFeatureAPIs(m, dv, cv)
	v, api, err := apis.Default()
	assert.Equal(t, v, dv)
	assert.Equal(t, api.(mockFeatureAPI).Length, 4)
	assert.Nil(t, err)

	v, api, err = apis.Current()
	assert.Equal(t, v, dv)
	assert.Equal(t, api.(mockFeatureAPI).Length, 4)
	assert.Nil(t, err)

	apis.Reset(v2, mockFeatureAPI{
		Length: 8,
	})
	api, err = apis.Get(v2)
	assert.Equal(t, api.(mockFeatureAPI).Length, 8)

	_, err = apis.Get(FeatureVersion("v3"))
	assert.NotNil(t, err)

	apis.SetCurrent(v2)
	v, api, err = apis.Current()
	assert.Equal(t, v, v2)
	assert.Equal(t, api.(mockFeatureAPI).Length, 8)
	assert.Nil(t, err)

	apis.Reset(v1, nil)
	api, err = apis.Get(v1)
	assert.Nil(t, api)
	assert.NotNil(t, err)
}
