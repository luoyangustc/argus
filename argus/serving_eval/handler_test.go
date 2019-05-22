package eval

import (
	"bytes"
	"context"
	"errors"
	"io"
	"io/ioutil"
	"testing"

	"github.com/stretchr/testify/assert"

	"qiniu.com/argus/atserving/model"
)

var _ Stream = buffer{}

type buffer struct {
	name string
	bs   []byte
}

func (buf buffer) Name() string { return buf.name }
func (buf buffer) Open(ctx context.Context) (io.ReadCloser, int64, error) {
	return ioutil.NopCloser(bytes.NewBuffer(buf.bs)), int64(len(buf.bs)), nil
}
func (buf buffer) Clean() error { return nil }

var _ Handler = &mockHandler{}

type mockHandler struct {
	resps   []EvalResponseInner
	streams []Stream

	err             error
	preEvalRet      []interface{}
	preGroupEvalRet [][]interface{}
}

func (mock *mockHandler) LoadEval(
	ctx context.Context, streams []Stream,
) ([]interface{}, error) {
	return mock.preEvalRet, mock.err
}
func (mock *mockHandler) LoadGroupEval(
	ctx context.Context, streams [][]Stream,
) ([][]interface{}, error) {
	return mock.preGroupEvalRet, mock.err
}
func (mock *mockHandler) PreEval(
	ctx context.Context, req model.EvalRequestInner,
) (model.EvalRequestInner, error) {
	return req, mock.err
}
func (mock *mockHandler) PreGroupEval(
	ctx context.Context, req model.GroupEvalRequestInner,
) (model.GroupEvalRequestInner, error) {
	return req, mock.err
}
func (mock *mockHandler) Eval(
	context.Context, []model.EvalRequestInner,
) ([]EvalResponseInner, error) {
	return mock.resps, mock.err
}
func (mock *mockHandler) GroupEval(
	context.Context, []model.GroupEvalRequestInner,
) ([]EvalResponseInner, error) {
	return mock.resps, mock.err
}

////////////////////////////////////////////////////////////////////////////////

func TestCopyFilePool(t *testing.T) {

	pool := newCopyFilePool(3)
	pool.Add(func() (context.Context, string, interface{}, error) {
		return context.Background(), "a", "1", nil
	})
	pool.Add(func() (context.Context, string, interface{}, error) {
		return context.Background(), "b", "2", nil
	})
	pool.Add(func() (context.Context, string, interface{}, error) {
		return context.Background(), "c", "", errors.New("error")
	})
	result := pool.Wait()
	assert.Equal(t, 3, len(result))
	assert.Equal(t, "1", result["a"].Ret)
	assert.Equal(t, "2", result["b"].Ret)
	assert.Equal(t, "error", result["c"].Error.Error())

}

func TestCopyFilePoolPanic(t *testing.T) {

	pool := newCopyFilePool(3)
	pool.Add(func() (context.Context, string, interface{}, error) {
		return context.Background(), "a", "1", nil
	})
	pool.Add(func() (context.Context, string, interface{}, error) {
		panic(errors.New("xxx"))
	})
	pool.Add(func() (context.Context, string, interface{}, error) {
		return context.Background(), "c", "", errors.New("error")
	})
	pool.Wait()
	assert.Equal(t, "xxx", pool.Err().Error())

}
