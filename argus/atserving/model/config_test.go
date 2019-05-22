package model

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestConfigKeyWorker(t *testing.T) {

	{
		var key ConfigKeyWorker
		(&key).Parse([]byte(KeyWorkerDefault))
		assert.Nil(t, key.App)
		assert.Nil(t, key.Version)
	}
	{
		var key ConfigKeyWorker
		(&key).Parse([]byte("/ava/serving/worker/app/foo/default"))
		assert.Equal(t, "foo", *key.App)
		assert.Nil(t, key.Version)
	}
	{
		var key ConfigKeyWorker
		(&key).Parse([]byte("/ava/serving/worker/app/foo/release/v1"))
		assert.Equal(t, "foo", *key.App)
		assert.Equal(t, "v1", *key.Version)
	}
	{
		var key ConfigKeyWorker
		(&key).Parse([]byte(""))
		assert.Error(t, ErrBadConfigKey)
	}

}

func TestConfigKeyAppMetadata(t *testing.T) {

	{
		var key ConfigKeyAppMetadata
		(&key).Parse([]byte("/ava/serving/app/metadata/foo"))
		assert.Equal(t, "foo", key.App)
	}
	{
		var key ConfigKeyAppMetadata
		(&key).Parse([]byte(""))
		assert.Error(t, ErrBadConfigKey)
	}

}

func TestConfigKeyAppRelease(t *testing.T) {

	{
		var key ConfigKeyAppRelease
		(&key).Parse([]byte("/ava/serving/app/release/foo/v1"))
		assert.Equal(t, "foo", key.App)
		assert.Equal(t, "v1", key.Version)
	}
	{
		var key ConfigKeyAppRelease
		(&key).Parse([]byte(""))
		assert.Error(t, ErrBadConfigKey)
	}

}
