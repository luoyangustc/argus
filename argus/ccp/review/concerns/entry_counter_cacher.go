package concerns

import (
	"sync"
	"time"

	xlog "github.com/qiniu/xlog.v1"
	"qiniu.com/argus/ccp/review/dao"
	"qiniu.com/argus/ccp/review/enums"
	"qiniu.com/argus/ccp/review/model"
)

var EntryCounterCacher *_EntryCounterCacher

type _EntryCounterCacher struct {
	counters map[string]*model.SetCounter
	changes  chan *model.SetCounter
	mux      *sync.Mutex
	logger   *xlog.Logger
	timeout  time.Duration // touch db with timeout
	done     chan struct{}
}

func NewEntryCounterCacher(xl *xlog.Logger, timeout time.Duration, size int) *_EntryCounterCacher {
	return &_EntryCounterCacher{
		counters: make(map[string]*model.SetCounter),
		changes:  make(chan *model.SetCounter, size),
		mux:      new(sync.Mutex),
		logger:   xl,
		timeout:  timeout,
		done:     make(chan struct{}),
	}
}

func (this *_EntryCounterCacher) Start() {
	// start channel first
	go func() {
		for {
			select {
			case c := <-this.changes:
				this.setOrMerge(c)
			case <-this.done:
				return
			}
		}
	}()

	go func() {
		ticker := time.NewTicker(this.timeout)
		for {
			select {
			case <-ticker.C: // maybe there is a mongodb connection leck promblem
				this.FlushDB()
			case <-this.done:
				this.FlushDB()
				return
			}
		}
	}()
}

func (this *_EntryCounterCacher) Close() {
	close(this.done)
}

func (this *_EntryCounterCacher) Add(c *model.SetCounter) {
	this.changes <- c
}

func (this *_EntryCounterCacher) setOrMerge(c *model.SetCounter) {
	this.mux.Lock()
	defer this.mux.Unlock()
	if counter, ok := this.counters[c.SetId]; ok {
		counter.MergeWith(c)
	} else {
		this.counters[c.SetId] = c
	}
}

func (this *_EntryCounterCacher) get(setId string) (c *model.SetCounter) {
	this.mux.Lock()
	defer this.mux.Unlock()
	if counter, ok := this.counters[setId]; ok {
		return counter
	}
	return nil
}

func (this *_EntryCounterCacher) FlushDB() {
	this.mux.Lock()
	defer this.mux.Unlock()

	if len(this.counters) == 0 {
		return
	}

	counters := this.counters
	this.counters = make(map[string]*model.SetCounter)
	go this.doSaveCounters(counters)
}

func (this *_EntryCounterCacher) doSaveCounters(counters map[string]*model.SetCounter) {
	if len(counters) == 0 {
		return
	}

	for _, c := range counters {
		// must have 2 times for try
		if ok := this.doSaveCounter(c); ok {
			continue
		}

		if ok := this.doSaveCounter(c); ok {
			continue
		}

		this.logger.Errorf("flush counter failed: %#v", c)
	}
}

func (this *_EntryCounterCacher) doSaveCounter(c *model.SetCounter) (ok bool) {
	counter, err := dao.SetCounterDAO.Find(nil, c.SetId)
	if err != nil && err != dao.ErrNotFound {
		this.logger.Errorf("dao.SetCounterDAO.Find(%s): %v", c.SetId, err)
		return
	}

	isNewRecord := (err == dao.ErrNotFound)

	// new record
	if isNewRecord {
		set, err2 := dao.EntrySetCache.MustGet(c.SetId)
		if err2 != nil {
			this.logger.Errorf("dao.EntrySetCache.MustGet(%s): %v", c.SetId, err)
			return
		}

		counter = &model.SetCounter{
			UserId:      set.Uid,
			ResourceId:  set.ResourceId(),
			SetId:       set.SetId,
			Values:      make(map[enums.Scene]int),
			LelfValues:  make(map[enums.Scene]int),
			Values2:     make(map[enums.Scene]int),
			LelfValues2: make(map[enums.Scene]int),
			Version:     1,
		}
	}

	counter.MergeWith(c)

	if isNewRecord {
		err = dao.SetCounterDAO.Insert(nil, counter)
	} else {
		// we need a luck lock for this logic
		oldVersion := counter.Version
		counter.Version += 1
		err = dao.SetCounterDAO.Update(nil, counter, oldVersion)
	}

	return err == nil
}
