package gate

import (
	"context"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestProducerTopic(t *testing.T) {

	var p = &producer{}
	assert.Equal(t, "first_foo", p.topic(context.Background(), "foo", nil))
	assert.Equal(t, "first_foo_v1", p.topic(context.Background(), "foo", sp("v1")))

}
