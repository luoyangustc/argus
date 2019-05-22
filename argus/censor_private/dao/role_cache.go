package dao

import (
	"sync"

	"qiniu.com/argus/censor_private/proto"
)

var (
	roleCacheLimit = 50
	RoleCache      = NewRoleCache()
)

type _RoleCache struct {
	cache map[string][]proto.Role
	*sync.Mutex
}

func NewRoleCache() *_RoleCache {
	return &_RoleCache{
		cache: make(map[string][]proto.Role, roleCacheLimit),
		Mutex: new(sync.Mutex),
	}
}

func (c *_RoleCache) Remove(userId string) {
	c.Lock()
	defer c.Unlock()
	delete(c.cache, userId)
}

func (c *_RoleCache) Get(userId string) []proto.Role {
	c.Lock()
	defer c.Unlock()
	if roles, ok := c.cache[userId]; ok {
		return roles
	}
	return nil
}

func (c *_RoleCache) Set(userId string, roles []proto.Role) {
	c.Lock()
	defer c.Unlock()
	delete(c.cache, userId)
	if len(c.cache) > roleCacheLimit {
		newCache := make(map[string][]proto.Role, roleCacheLimit)
		newCache[userId] = roles
		c.cache = newCache
	} else {
		c.cache[userId] = roles
	}
}
