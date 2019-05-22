package batch

import (
	"context"
	"errors"
	"fmt"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
)

func TestBatch(t *testing.T) {
	e := New().Attach(func(ctx context.Context) error {
		time.Sleep(time.Second)
		return nil
	}).Attach(func(ctx context.Context) error {
		time.Sleep(500 * time.Millisecond)
		return nil
	}).Attach(func(ctx context.Context) error {
		time.Sleep(800 * time.Millisecond)
		return nil
	}).Run(context.Background())
	assert.NoError(t, e)

	e = New().Attach(func(ctx context.Context) error {
		time.Sleep(time.Second)
		return errors.New("error 3")
	}).Attach(func(ctx context.Context) error {
		time.Sleep(500 * time.Millisecond)
		return nil
	}).Attach(func(ctx context.Context) error {
		time.Sleep(300 * time.Millisecond)
		return errors.New("error 1")
	}).Attach(func(ctx context.Context) error {
		time.Sleep(800 * time.Millisecond)
		return errors.New("error 2")
	}).Run(context.Background())
	assert.Error(t, e, "error 1")
}

func ExampleBatch() {
	sleep := func(ms int) Task {
		return func(context.Context) error {
			time.Sleep(time.Duration(ms) * time.Millisecond)
			fmt.Println(ms)
			return nil
		}
	}

	New().
		Attach(sleep(100)).
		Attach(sleep(700)).
		Attach(sleep(500)).
		Attach(sleep(300)).
		Run(context.Background())

	// Output:
	// 100
	// 300
	// 500
	// 700
}
