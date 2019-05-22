package sts

import (
	"bytes"
	"context"
	"io/ioutil"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"

	URI "qiniu.com/argus/sts/uri"
)

func TestSlotIO(t *testing.T) {

	slot := _newSlot(newMemory(), nil)

	var (
		uri    = URI.NewURI(time.Now().Format("20060102150405"))
		bs     = []byte("xxxxxxxxxxxxxxxxx")
		length = int64(len(bs))
	)

	func() {
		_, _, err := slot.Get(context.Background(), uri, &length, true)
		assert.Equal(t, err, ErrNotExist)
	}()
	func() {
		err := slot.Post(context.Background(), uri, length, bytes.NewBuffer(bs))
		assert.NoError(t, err, "post data")
	}()
	func() {
		reader, _, err := slot.Get(context.Background(), uri, &length, true)
		assert.NoError(t, err, "get reader")
		defer reader.Close()
		bs2, _ := ioutil.ReadAll(reader)
		assert.Equal(t, len(bs), len(bs2))
	}()
	func() {
		reader, length2, err := slot.Get(context.Background(), uri, nil, true)
		assert.NoError(t, err, "get reader")
		defer reader.Close()
		bs2, _ := ioutil.ReadAll(reader)
		assert.Equal(t, len(bs), len(bs2))
		assert.Equal(t, len(bs), int(length2))
	}()
}

func TestStorageIO(t *testing.T) {

	stg, _ := NewStorage(
		StorageConfig{Overdue: time.Minute},
		func(ctx context.Context) ([]Storage, []time.Time, error) {
			return []Storage{_newSlot(newMemory(), nil)}, []time.Time{time.Now()}, nil
		},
		func(ctx context.Context, now time.Time) (Storage, error) {
			return _newSlot(newMemory(), nil), nil
		},
	)

	var (
		uri    = URI.NewURI(time.Now().Format("20060102150405"))
		bs     = []byte("xxxxxxxxxxxxxxxxx")
		length = int64(len(bs))
	)

	func() {
		_, _, err := stg.Get(context.Background(), uri, &length, true)
		assert.Equal(t, err, ErrNotExist)
	}()
	func() {
		err := stg.Post(context.Background(), uri, length, bytes.NewBuffer(bs))
		assert.NoError(t, err, "post data")
	}()
	func() {
		reader, _, err := stg.Get(context.Background(), uri, &length, true)
		assert.NoError(t, err, "get reader")
		defer reader.Close()
		bs2, _ := ioutil.ReadAll(reader)
		assert.Equal(t, len(bs), len(bs2))
	}()

}
