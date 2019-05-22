package dao

import (
	"sync"

	"qiniu.com/argus/censor_private/proto"
)

var (
	setCacheLimit = 1000
	SetCache      = NewSetCache()
)

type _SetCache struct {
	cache map[string]*proto.Set
	*sync.Mutex
}

func NewSetCache() *_SetCache {
	return &_SetCache{
		cache: make(map[string]*proto.Set, setCacheLimit),
		Mutex: new(sync.Mutex),
	}
}

func (c *_SetCache) MustGet(setId string) (*proto.Set, error) {
	if set, ok := c.cache[setId]; ok {
		return set, nil
	}

	set, err := SetDao.Find(setId)
	if err != nil {
		return nil, err
	}

	c.MustSet(set)
	return set, nil
}

func (c *_SetCache) MustSet(set *proto.Set) {
	c.Lock()
	defer c.Unlock()

	id := set.Id
	if len(c.cache) > setCacheLimit {
		newCache := make(map[string]*proto.Set, setCacheLimit)
		newCache[id] = set
		c.cache = newCache
	} else {
		c.cache[id] = set
	}
}
