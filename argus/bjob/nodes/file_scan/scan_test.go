package file_scan

import (
	"bytes"
	"context"
	"io"
	"io/ioutil"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestIter(t *testing.T) {

	{
		iter := EntryIter{RS: []func() (io.ReadCloser, error){}}
		var count int
		for {
			_, ok := iter.Next(context.Background())
			if !ok {
				break
			}
			count++
		}
		assert.Equal(t, 0, count)
	}

	{
		iter := EntryIter{RS: []func() (io.ReadCloser, error){
			func() (io.ReadCloser, error) {
				return ioutil.NopCloser(bytes.NewReader(
					[]byte(`xxxx
					xxxx
					xxxx
					xxxx`))), nil
			},
			func() (io.ReadCloser, error) {
				return ioutil.NopCloser(bytes.NewReader(
					[]byte(`yyyy
					yyyy`))), nil
			},
			func() (io.ReadCloser, error) {
				return ioutil.NopCloser(bytes.NewReader(
					[]byte(`yyyy
					`))), nil
			},
		}}
		var count int
		for {
			_, ok := iter.Next(context.Background())
			if !ok {
				break
			}
			count++
		}
		assert.Equal(t, 7, count)
	}

}
