package util

import (
	"context"
	"errors"
	"fmt"
	"time"
)

func AutoWait(ctx context.Context, param interface{}, f func(context.Context, interface{}) bool, interval time.Duration) error {
	for {
		select {
		case <-ctx.Done():
			fmt.Println("timeout!")
			return errors.New("Timeout")
		default:
			status := f(ctx, param)
			if status {
				fmt.Println("finish!")
				return nil
			}
			time.Sleep(interval)
		}
	}
}
