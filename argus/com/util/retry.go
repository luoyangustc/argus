package util

import (
	"context"
	"fmt"
	"time"
)

// Retry calls the function with binary exponential backoff.
func Retry(ctx context.Context, attempts int, dur time.Duration, f func() error) (err error) {
	select {
	case <-ctx.Done():
		return ctx.Err()
	default:
	}

	defer func() {
		if r := recover(); r != nil {
			err = fmt.Errorf("panic: %v", r)
		}
	}()

	if e := f(); e != nil {
		if attempts--; attempts > 0 {
			time.Sleep(dur)
			return Retry(ctx, attempts, 2*dur, f)
		}
		return e
	}
	return nil
}
