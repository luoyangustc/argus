package sts

import (
	"context"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sync"
	"time"

	"github.com/qiniu/xlog.v1"
	URI "qiniu.com/argus/sts/uri"
)

// Storage ...
type Storage interface {
	Fetch(ctx context.Context, uri URI.Uri, length *int64, sync bool) error
	Get(
		ctx context.Context, uri URI.Uri, length *int64, justLocal bool,
	) (io.ReadCloser, int64, error)
	Post(ctx context.Context, uri URI.Uri, length int64, reader io.Reader) error
	Open(ctx context.Context, uri URI.Uri, length int64) (io.WriteCloser, error)
	RemoveAll(ctx context.Context) error
}

//----------------------------------------------------------------------------//

var _ Storage = &storage{}

// StorageConfig ...
type StorageConfig struct {
	Dir     string        `json:"dir"`
	Overdue time.Duration `json:"overdue"`
}

// InitSlots ...
func (c StorageConfig) InitSlots(
	fetcher Fetcher,
) func(context.Context) ([]Storage, []time.Time, error) {

	return func(ctx context.Context) ([]Storage, []time.Time, error) {
		xl := xlog.FromContextSafe(ctx)
		dir, err := os.Open(c.Dir)
		if err != nil {
			xl.Errorf("open dir failed. %s %v", c.Dir, err)
			return nil, nil, err
		}
		defer dir.Close()

		infos, err := dir.Readdir(0)
		if err != nil {
			xl.Errorf("readdir failed. %v", err)
			return nil, nil, err
		}

		var (
			slots = make([]Storage, 0, len(infos))
			times = make([]time.Time, 0, len(infos))
		)
		for _, info := range infos {
			if info.IsDir() {
				t, err := time.Parse("20060102150405", info.Name())
				if err != nil {
					continue
				}
				slot, err := newSlot(ctx, filepath.Join(c.Dir, info.Name()), fetcher)
				if err != nil {
					return nil, nil, err
				}
				slots = append(slots, slot)
				times = append(times, t)
			}
		}
		return slots, times, nil
	}
}

// NewSlot ...
func (c StorageConfig) NewSlot(fetcher Fetcher) func(context.Context, time.Time) (Storage, error) {
	return func(ctx context.Context, now time.Time) (Storage, error) {
		return newSlot(ctx, filepath.Join(c.Dir, now.Format("20060102150405")), fetcher)
	}
}

type storage struct {
	slots []Storage
	times []time.Time
	*sync.RWMutex

	newSlot func(context.Context, time.Time) (Storage, error)
}

// NewStorage ...
func NewStorage(
	config StorageConfig,
	initSlots func(context.Context) ([]Storage, []time.Time, error),
	newSlot func(context.Context, time.Time) (Storage, error),
) (Storage, error) {

	ctx := xlog.NewContextWith(context.Background(), "NewStorage")

	slots, times, err := initSlots(ctx)
	if err != nil {
		return nil, err
	}
	if slots == nil {
		slots = make([]Storage, 0)
		times = make([]time.Time, 0)
	}
	if len(slots) == 0 {
		now := time.Now()
		slot, err := newSlot(ctx, now)
		if err != nil {
			return nil, err
		}
		slots = append(slots, slot)
		times = append(times, now)
	}

	stg := &storage{
		slots:   slots,
		times:   times,
		RWMutex: new(sync.RWMutex),
		newSlot: newSlot,
	}

	go func() {
		for _ = range time.Tick(config.Overdue / 2) {
			func() {
				xl := xlog.NewDummy()
				ctx := xlog.NewContext(context.Background(), xl)
				defer func() {
					if err := recover(); err != nil {
						xl.Errorf("rotate panic. %s", err)
					}
				}()
				_ = stg.rotate(ctx, config.Overdue)
			}()
		}
	}()

	return stg, nil
}

func (s *storage) rotate(ctx context.Context, duration time.Duration) error {
	var (
		now      = time.Now()
		last     = now.Add(-1 * duration)
		newSlots = []Storage{}
		newTimes = []time.Time{}
		rmSlots  = make([]Storage, 0)
	)

	slot1, err := s.newSlot(ctx, now)
	if err != nil {
		return err
	}
	newSlots = append(newSlots, slot1)
	newTimes = append(newTimes, now)

	waitForLock("storage", "", "rotate").Inc()
	s.Lock()
	for i := range s.slots {
		if s.times[i].Before(last) {
			rmSlots = append(rmSlots, s.slots[i])
		} else {
			newSlots = append(newSlots, s.slots[i])
			newTimes = append(newTimes, s.times[i])
		}
	}
	s.slots = newSlots
	s.times = newTimes
	s.Unlock()
	waitForLock("storage", "", "rotate").Dec()

	for _, slot := range rmSlots {
		slot.RemoveAll(ctx)
	}
	return nil
}

func (s *storage) Fetch(ctx context.Context, uri URI.Uri, length *int64, sync bool) error {
	var (
		xl  = xlog.FromContextSafe(ctx)
		stg Storage
	)
	begin := time.Now()
	waitForLock("storage", "", "fetch").Inc()
	s.RLock()
	xl.Info(time.Since(begin))
	stg = s.slots[0]
	s.RUnlock()
	waitForLock("storage", "", "fetch").Dec()
	return stg.Fetch(ctx, uri, length, sync)
}

func (s *storage) Get(ctx context.Context, uri URI.Uri, length *int64, justLocal bool) (io.ReadCloser, int64, error) {
	var stg Storage
	waitForLock("storage", "", "get").Inc()
	s.RLock()
	slots := s.slots
	s.RUnlock()
	waitForLock("storage", "", "get").Dec()
	for i, len := 0, len(slots); i < len-1; i++ {
		if reader, length2, err := slots[len-i-1].Get(ctx, uri, length, true); err != ErrNotExist {
			return reader, length2, err
		}
	}
	stg = slots[0]
	return stg.Get(ctx, uri, length, justLocal)
}

func (s *storage) Post(ctx context.Context, uri URI.Uri, length int64, reader io.Reader) error {
	var stg Storage
	waitForLock("storage", "", "post").Inc()
	s.RLock()
	stg = s.slots[0]
	s.RUnlock()
	waitForLock("storage", "", "post").Dec()
	return stg.Post(ctx, uri, length, reader)
}
func (s *storage) Open(ctx context.Context, uri URI.Uri, length int64) (io.WriteCloser, error) {
	var stg Storage
	waitForLock("storage", "", "open").Inc()
	s.RLock()
	stg = s.slots[0]
	s.RUnlock()
	waitForLock("storage", "", "open").Dec()
	return stg.Open(ctx, uri, length)
}

func (s *storage) RemoveAll(ctx context.Context) error {
	return nil // NOTHING TO DO
}

//----------------------------------------------------------------------------//

var _ Storage = &slot{}

type slot struct {
	Fetcher
	disk  Disk
	files map[URI.Uri]File
	*sync.Mutex
}

func _newSlot(disk Disk, fetcher Fetcher) *slot {
	return &slot{
		Fetcher: fetcher,
		disk:    disk,
		files:   make(map[URI.Uri]File),
		Mutex:   new(sync.Mutex),
	}
}

func newSlot(ctx context.Context, dir string, fetcher Fetcher) (*slot, error) {
	disk, err := newDisk(ctx, dir)
	if err != nil {
		return nil, err
	}
	return _newSlot(disk, fetcher), nil
}

func (s *slot) releaseFile(ctx context.Context, uri URI.Uri, check func() bool) {
	waitForLock("slot", fmt.Sprintf("%p", s.Mutex), "releaseFile").Inc()
	s.Lock()
	defer func() {
		s.Unlock()
		waitForLock("slot", fmt.Sprintf("%p", s.Mutex), "releaseFile").Dec()
	}()
	if check() {
		delete(s.files, uri)
	}
}

func (s *slot) Fetch(ctx context.Context, uri URI.Uri, length *int64, sync bool) error {
	var (
		xl   = xlog.FromContextSafe(ctx)
		file File
		ok   bool
	)
	begin := time.Now()
	waitForLock("slot", fmt.Sprintf("%p", s.Mutex), "fetch").Inc()
	s.Lock()
	xl.Info(time.Since(begin))
	if file, ok = s.files[uri]; !ok {
		file = newFile(uri, length, s.disk, func(check func() bool) { s.releaseFile(ctx, uri, check) })
		s.files[uri] = file
	}
	file.Acquire()
	s.Unlock()
	waitForLock("slot", fmt.Sprintf("%p", s.Mutex), "fetch").Dec()
	if sync {
		defer file.Release()
		_ = s.Fetcher.Fetch(ctx, uri, length, file)
	} else {
		go func() {
			defer file.Release()
			_ = s.Fetcher.Fetch(ctx, uri, length, file)
		}()
	}
	return nil
}

func (s *slot) Get(ctx context.Context, uri URI.Uri, length *int64, justLocal bool) (io.ReadCloser, int64, error) {
	var (
		file File
		ok   bool
	)
	waitForLock("slot", fmt.Sprintf("%p", s.Mutex), "get").Inc()
	s.Lock()
	if file, ok = s.files[uri]; !ok {
		file = newFile(uri, length, s.disk, func(check func() bool) { s.releaseFile(ctx, uri, check) })
		s.files[uri] = file
	}
	file.Acquire()
	s.Unlock()
	waitForLock("slot", fmt.Sprintf("%p", s.Mutex), "get").Dec()
	defer file.Release()

	reader, err := file.OpenAsReader(ctx)
	if err != ErrNotExist {
		return reader, file.GetLength(), err
	}
	if !justLocal && needFetch(uri) {
		err = s.Fetcher.Fetch(ctx, uri, length, file)
		if err == os.ErrExist {
			err = nil
		}
		if err != nil {
			return nil, 0, err
		}
	}
	if err != nil {
		return nil, 0, err
	}
	reader, err = file.OpenAsReader(ctx)
	return reader, file.GetLength(), err
}

func (s *slot) Post(ctx context.Context, uri URI.Uri, length int64, reader io.Reader) error {
	var (
		file File
		ok   bool
	)
	waitForLock("slot", fmt.Sprintf("%p", s.Mutex), "post").Inc()
	s.Lock()
	if file, ok = s.files[uri]; !ok {
		file = newFile(uri, &length, s.disk, func(check func() bool) { s.releaseFile(ctx, uri, check) })
		s.files[uri] = file
	}
	file.Acquire()
	s.Unlock()
	waitForLock("slot", fmt.Sprintf("%p", s.Mutex), "post").Dec()
	defer file.Release()
	_, err := file.Write(ctx, reader)
	return err
}

func (s *slot) Open(ctx context.Context, uri URI.Uri, length int64) (io.WriteCloser, error) {
	var (
		file File
		ok   bool
	)
	waitForLock("slot", fmt.Sprintf("%p", s.Mutex), "open").Inc()
	s.Lock()
	if file, ok = s.files[uri]; !ok {
		file = newFile(uri, &length, s.disk, func(check func() bool) { s.releaseFile(ctx, uri, check) })
		s.files[uri] = file
	}
	file.Acquire()
	s.Unlock()
	waitForLock("slot", fmt.Sprintf("%p", s.Mutex), "open").Dec()
	// defer file.Release()
	return file.OpenAsWriter(ctx, &length)
}

func (s *slot) RemoveAll(ctx context.Context) error {
	return s.disk.RemoveAll(ctx)
}
