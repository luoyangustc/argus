package batch

import (
	"context"
	"errors"
	"runtime/debug"

	log "qiniupkg.com/x/log.v7"
)

type Task func(context.Context) error

// Batch runs multiple tasks concurrently
type Batch struct {
	ts []Task
}

func New() *Batch {
	return &Batch{
		ts: make([]Task, 0),
	}
}

func (b *Batch) Attach(task Task) *Batch {
	b.ts = append(b.ts, task)
	return b
}

// Run all tasks within the batch and returns when:
// - all tasks done without any error returned, or
// - first error returned by one of the tasks
func (b *Batch) Run(ctx context.Context) error {
	errs := make(chan error)

	for _, task := range b.ts {
		go func(task Task) {
			defer func() {
				if r := recover(); r != nil {
					errs <- errors.New("panic during batch tasks running: " + string(debug.Stack()))
				}
			}()

			errs <- task(ctx) // push error or nil to receiver
		}(task)
	}

	for i := len(b.ts); i > 0; i-- {
		select {
		case <-ctx.Done():
			return errors.New("Batch.Run() cancelled")
		case e := <-errs:
			if e != nil {
				go func() {
					for j := i - 1; j > 0; j++ {
						if e := <-errs; e != nil {
							log.Error("error after Batch.Run() returned:", e)
						}
					}
					log.Warn("all tasks done after Batch.Run() returned")
				}()
				return e
			}
		}
	}
	return nil
}
