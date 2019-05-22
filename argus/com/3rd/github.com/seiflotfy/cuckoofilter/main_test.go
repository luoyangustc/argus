package cuckoofilter

import (
	"fmt"
	"math/rand"
	"runtime"
	"testing"
	"time"
)

const letterBytes = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"

func bToMb(b uint64) uint64 {
	return b / 1024 / 1024
}

func printMemUsage() {
	var m runtime.MemStats
	for {
		runtime.ReadMemStats(&m)
		// For info on each, see: https://golang.org/pkg/runtime/#MemStats
		fmt.Printf("Alloc = %v MiB", bToMb(m.Alloc))
		fmt.Printf("\tTotalAlloc = %v MiB", bToMb(m.TotalAlloc))
		fmt.Printf("\tSys = %v MiB", bToMb(m.Sys))
		fmt.Printf("\tNumGC = %v\n", m.NumGC)
		time.Sleep(time.Second)
	}
}

func randStringBytes(n int) []byte {
	b := make([]byte, n)
	for i := range b {
		b[i] = letterBytes[rand.Intn(len(letterBytes))]
	}
	return b
}

func TestMaxConflictRate(t *testing.T) {
	total := 15 * 1000 * 1000

	cf, realCapacity := NewCuckooFilter(uint(total))
	fmt.Printf("real capacity is : %d \n", int(realCapacity))
	go printMemUsage()
	for i := 0; i < total; i++ {
		cf.InsertUnique(randStringBytes(20))
	}
	rate := (1 - float32(cf.Count())/float32(total)) * 10000
	fmt.Printf("capacity: %d, total: %d, count: %d, missing-percent: %f%%%%\n", realCapacity, total, cf.Count(), rate)
	if rate != 0 {
		t.Error("rate must be 0")
	}
}
