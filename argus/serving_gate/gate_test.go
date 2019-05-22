package gate

import (
	"testing"
	"time"

	"github.com/stretchr/testify.v2/assert"

	"qiniu.com/argus/atserving/model"
)

func TestGateEval(t *testing.T) {}

func TestEvals(t *testing.T) {
	e := NewEvals()

	{
		assert.False(t, e.IsAllowable(1, "a"))
		assert.False(t, e.IsAllowable(1, "b"))
		assert.False(t, e.IsAllowable(1, "c"))
		assert.False(t, e.IsAllowable(1, "d"))
		assert.False(t, e.IsAllowable(2, "a"))

		e.SetAppMetadataDefault(model.ConfigAppMetadata{Public: true})

		assert.True(t, e.IsAllowable(1, "a"))
		assert.True(t, e.IsAllowable(1, "b"))
		assert.True(t, e.IsAllowable(1, "c"))
		assert.True(t, e.IsAllowable(1, "d"))
		assert.True(t, e.IsAllowable(2, "a"))

		e.SetAppMetadata("a", model.ConfigAppMetadata{Public: false, UserWhiteList: []uint32{1}})

		assert.True(t, e.IsAllowable(1, "a"))
		assert.True(t, e.IsAllowable(1, "b"))
		assert.True(t, e.IsAllowable(1, "c"))
		assert.True(t, e.IsAllowable(1, "d"))
		assert.False(t, e.IsAllowable(2, "a"))

		e.SetAppMetadataDefault(model.ConfigAppMetadata{Public: false, UserWhiteList: []uint32{2}})
		e.SetAppMetadata("a", model.ConfigAppMetadata{Public: false})
		e.SetAppMetadata("b", model.ConfigAppMetadata{Public: true})
		e.SetAppMetadata("c", model.ConfigAppMetadata{Public: false, Owner: model.Owner{UID: 1}})
		e.SetAppMetadata("d", model.ConfigAppMetadata{Public: false, UserWhiteList: []uint32{1}})

		assert.False(t, e.IsAllowable(1, "a"))
		assert.True(t, e.IsAllowable(1, "b"))
		assert.True(t, e.IsAllowable(1, "c"))
		assert.True(t, e.IsAllowable(1, "d"))
		assert.True(t, e.IsAllowable(2, "a"))

	}

	assert.False(t, e.Available("a", nil))
	assert.False(t, e.Available("a", sp("1")))
	assert.Equal(t, time.Second*10, e.Timeout("a", nil))
	assert.Equal(t, time.Second*10, e.Timeout("a", sp("1")))

	e.SetWorkerDefault(model.ConfigWorker{Timeout: time.Second * 11})
	e.SetWorker("a", nil, model.ConfigWorker{Timeout: time.Second * 12})
	e.SetWorker("b", sp("1"), model.ConfigWorker{Timeout: time.Second * 12})
	e.SetWorker("c", nil, model.ConfigWorker{Timeout: time.Second * 12})
	e.SetWorker("c", sp("1"), model.ConfigWorker{Timeout: time.Second * 13})
	e.SetWorker("d", nil, model.ConfigWorker{Timeout: time.Second * 12})
	e.SetWorker("d", sp("1"), model.ConfigWorker{Timeout: time.Second * 13})

	e.Register("d", "1", model.ConfigAppRelease{})
	e.Register("e", "1", model.ConfigAppRelease{})
	e.Register("e", "2", model.ConfigAppRelease{})

	assert.False(t, e.Available("a", nil))
	assert.False(t, e.Available("b", sp("1")))
	assert.True(t, e.Available("d", nil))
	assert.True(t, e.Available("d", sp("1")))
	assert.False(t, e.Available("d", sp("2")))
	assert.True(t, e.Available("e", nil))
	assert.True(t, e.Available("e", sp("1")))
	assert.True(t, e.Available("e", sp("2")))

	assert.Equal(t, time.Second*12, e.Timeout("a", nil))
	assert.Equal(t, time.Second*12, e.Timeout("a", sp("1")))
	assert.Equal(t, time.Second*11, e.Timeout("b", nil))
	assert.Equal(t, time.Second*12, e.Timeout("b", sp("1")))
	assert.Equal(t, time.Second*12, e.Timeout("c", nil))
	assert.Equal(t, time.Second*13, e.Timeout("c", sp("1")))
	assert.Equal(t, time.Second*12, e.Timeout("c", sp("2")))
	assert.Equal(t, time.Second*12, e.Timeout("d", nil))
	assert.Equal(t, time.Second*13, e.Timeout("d", sp("1")))
	assert.Equal(t, time.Second*12, e.Timeout("d", sp("2")))
	assert.Equal(t, time.Second*11, e.Timeout("e", nil))

	e.UnsetWorker("a", nil)
	e.UnsetWorker("b", sp("1"))
	e.UnsetWorker("c", sp("1"))
	e.UnsetWorker("d", nil)

	e.Unregister("d", "1")
	e.Unregister("e", "1")

	assert.False(t, e.Available("a", nil))
	assert.False(t, e.Available("b", sp("1")))
	assert.False(t, e.Available("d", nil))
	assert.False(t, e.Available("d", sp("1")))
	assert.False(t, e.Available("d", sp("2")))
	assert.True(t, e.Available("e", nil))
	assert.False(t, e.Available("e", sp("1")))
	assert.True(t, e.Available("e", sp("2")))

	assert.Equal(t, time.Second*11, e.Timeout("a", nil))
	assert.Equal(t, time.Second*11, e.Timeout("a", sp("1")))
	assert.Equal(t, time.Second*11, e.Timeout("b", nil))
	assert.Equal(t, time.Second*11, e.Timeout("b", sp("1")))
	assert.Equal(t, time.Second*12, e.Timeout("c", nil))
	assert.Equal(t, time.Second*12, e.Timeout("c", sp("1")))
	assert.Equal(t, time.Second*12, e.Timeout("c", sp("2")))
	assert.Equal(t, time.Second*11, e.Timeout("d", nil))
	assert.Equal(t, time.Second*13, e.Timeout("d", sp("1")))
	assert.Equal(t, time.Second*11, e.Timeout("d", sp("2")))
	assert.Equal(t, time.Second*11, e.Timeout("e", nil))

}
