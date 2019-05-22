package vframe

import (
	"context"
	"sync"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestStopProducer(t *testing.T) {

	job := newProducerJob(nil)
	wg := sync.WaitGroup{}
	wg.Add(1)
	go func() {
		defer wg.Done()
		job.end(context.Background(), "", EndResponse{})
	}()
	job.Stop()

	wg.Wait()
	<-job.ch
	assert.True(t, true)

}
