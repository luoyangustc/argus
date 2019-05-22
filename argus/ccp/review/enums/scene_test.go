package enums

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestSceneIsValid(t *testing.T) {
	assertion := assert.New(t)

	assertion.True(ScenePulp.IsValid())
	assertion.True(SceneTerror.IsValid())
	assertion.True(ScenePolitician.IsValid())
	assertion.False(Scene("pass").IsValid())
}
