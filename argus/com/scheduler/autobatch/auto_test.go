package autobatch

import (
	"context"
	"errors"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
)

func TestTasks(t *testing.T) {

	t.Run("1", func(t *testing.T) {
		tasks := newTasks()
		assert.False(t, tasks.Has(&_Task{ID: 1}))
		assert.False(t, tasks.Pick(&_Task{ID: 1}))
		tasks.Push(&_Task{ID: 1})
		assert.True(t, tasks.Has(&_Task{ID: 1}))
		assert.True(t, tasks.Pick(&_Task{ID: 1}))
		assert.False(t, tasks.Has(&_Task{ID: 1}))
		assert.False(t, tasks.Pick(&_Task{ID: 1}))
	})
	t.Run("2", func(t *testing.T) {
		tasks := newTasks()
		tasks.Push(&_Task{ID: 1})
		tasks.Push(&_Task{ID: 2})
		tasks.Push(&_Task{ID: 3})
		tasks.Push(&_Task{ID: 4})
		tasks.Push(&_Task{ID: 5})

		assert.True(t, tasks.Pick(&_Task{ID: 5}))
		assert.True(t, tasks.Pick(&_Task{ID: 3}))
		assert.True(t, tasks.Pick(&_Task{ID: 1}))

		assert.False(t, tasks.Has(&_Task{ID: 1}))
		assert.True(t, tasks.Has(&_Task{ID: 2}))
		assert.False(t, tasks.Has(&_Task{ID: 3}))
		assert.True(t, tasks.Has(&_Task{ID: 4}))
		assert.False(t, tasks.Has(&_Task{ID: 5}))
	})
	t.Run("3", func(t *testing.T) {
		tasks := newTasks()
		t1 := &_Task{ID: 1}
		t2 := &_Task{ID: 2}
		t3 := &_Task{ID: 3}
		t4 := &_Task{ID: 4}
		t5 := &_Task{ID: 5}
		tasks.Push(t1)
		tasks.Push(t2)
		tasks.Push(t3)
		tasks.Push(t4)
		tasks.Push(t5)

		tasks.Pick(&_Task{ID: 1})
		t3.Cancel()
		tasks.Pick(&_Task{ID: 5})

		ts := tasks.PopN(3, []*_Task{})
		assert.Equal(t, 2, len(ts))
		assert.Equal(t, int64(2), ts[0].ID)
		assert.Equal(t, int64(4), ts[1].ID)
	})
}

func TestBatch(t *testing.T) {

	t.Run("1", func(t *testing.T) {
		var f0 BatchFunc
		f := func(ctx context.Context, params []interface{}) ([]interface{}, error) { return f0(ctx, params) }
		b := NewBatch(BatchConfig{}, f)

		f0 = func(ctx context.Context, params []interface{}) ([]interface{}, error) {
			assert.Equal(t, 1, len(params))
			assert.Equal(t, 1, params[0].(int))
			return []interface{}{true}, nil
		}
		ret, err := b.Do(context.Background(), 1)
		assert.NoError(t, err)
		assert.Equal(t, true, ret.(bool))

		err0 := errors.New("")
		f0 = func(ctx context.Context, params []interface{}) ([]interface{}, error) { return nil, err0 }
		_, err = b.Do(context.Background(), 2)
		assert.Error(t, err0, err)
	})

	t.Run("2", func(t *testing.T) {
		var count uint32
		f := func(ctx context.Context, params []interface{}) ([]interface{}, error) {
			t.Logf("%d\n", len(params))
			time.Sleep(time.Millisecond * 10)
			atomic.AddUint32(&count, uint32(len(params)))
			ret := make([]interface{}, 0, len(params))
			for _, param := range params {
				ret = append(ret, 1+param.(int))
			}
			return ret, nil
		}
		b := NewBatch(BatchConfig{MaxBatchSize: 5}, f)

		var wg = sync.WaitGroup{}
		for i := 0; i < 100; i++ {
			time.Sleep(time.Millisecond)
			wg.Add(1)
			go func(index int) {
				defer wg.Done()
				ret, err := b.Do(context.Background(), index)
				assert.NoError(t, err)
				assert.Equal(t, index+1, ret.(int))
			}(i)
		}
		wg.Wait()
		assert.Equal(t, uint32(100), count)
	})

}
