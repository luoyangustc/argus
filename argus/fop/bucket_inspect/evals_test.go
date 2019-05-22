package bucket_inspect

import (
	"context"
	"io"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestBucketInspect(t *testing.T) {

	var dis bool
	var err error
	{
		_, dis, err = qpulp(context.Background(), nil, nil)
		assert.Error(t, err)
		assert.Equal(t, dis, false)

		_, dis, err = qpulp(context.Background(), []string{"v1"}, &struct {
			ReqBody io.ReadCloser
			Cmd     string `json:"cmd"`
			URL     string `json:"url"`
		}{
			ReqBody: &MockReader{
				Len: 10,
			},
		})
		assert.NoError(t, err)
		assert.Equal(t, dis, false)

		_, dis, err = qpulp(context.Background(), []string{"v2"}, &struct {
			ReqBody io.ReadCloser
			Cmd     string `json:"cmd"`
			URL     string `json:"url"`
		}{
			ReqBody: &MockReader{
				Len: 10,
			},
		})
		assert.Error(t, err)
		assert.Equal(t, dis, false)
	}

	{
		_, dis, err = imageCensor(context.Background(), nil, nil)
		assert.Error(t, err)
		assert.Equal(t, dis, false)

		_, dis, err = imageCensor(context.Background(), []string{"v1"}, &struct {
			ReqBody io.ReadCloser
			Cmd     string `json:"cmd"`
			URL     string `json:"url"`
		}{
			ReqBody: &MockReader{
				Len: 10,
			},
		})
		assert.NoError(t, err)
		assert.Equal(t, dis, false)

		_, dis, err = imageCensor(context.Background(), []string{"v2"}, &struct {
			ReqBody io.ReadCloser
			Cmd     string `json:"cmd"`
			URL     string `json:"url"`
		}{
			ReqBody: &MockReader{
				Len: 10,
			},
		})
		assert.NoError(t, err)
		assert.Equal(t, dis, false)

		_, dis, err = imageCensor(context.Background(), []string{"v3"}, &struct {
			ReqBody io.ReadCloser
			Cmd     string `json:"cmd"`
			URL     string `json:"url"`
		}{
			ReqBody: &MockReader{
				Len: 10,
			},
		})
		assert.Error(t, err)
		assert.Equal(t, dis, false)
	}

	{
		_, dis, err = videoCensor(context.Background(), nil, nil)
		assert.Error(t, err)
		assert.Equal(t, dis, false)

		_, dis, err = videoCensor(context.Background(), []string{"v1"}, &struct {
			ReqBody io.ReadCloser
			Cmd     string `json:"cmd"`
			URL     string `json:"url"`
		}{
			ReqBody: &MockReader{
				Len: 10,
			},
		})
		assert.Error(t, err)
		assert.Equal(t, dis, false)

		_, dis, err = videoCensor(context.Background(), []string{"v2"}, &struct {
			ReqBody io.ReadCloser
			Cmd     string `json:"cmd"`
			URL     string `json:"url"`
		}{
			ReqBody: &MockReader{
				Len: 10,
			},
		})
		assert.NoError(t, err)
		assert.Equal(t, dis, false)

		_, dis, err = videoCensor(context.Background(), []string{"v3"}, &struct {
			ReqBody io.ReadCloser
			Cmd     string `json:"cmd"`
			URL     string `json:"url"`
		}{
			ReqBody: &MockReader{
				Len: 10,
			},
		})
		assert.Error(t, err)
		assert.Equal(t, dis, false)
	}

}
