package dao

import (
	"sync"

	"qiniu.com/argus/ccp/review/enums"
	"qiniu.com/argus/ccp/review/model"
)

var (
	entrySetCacheLimit = 10000
	EntrySetCache      = NewEntrySetCache()
)

type _EntrySetCache struct {
	cache map[string]*model.Set
	*sync.Mutex
}

func NewEntrySetCache() *_EntrySetCache {
	return &_EntrySetCache{
		cache: make(map[string]*model.Set, entrySetCacheLimit),
		Mutex: new(sync.Mutex),
	}
}

func (this *_EntrySetCache) GetDao(setId string) (EntryDAO, error) {
	set, err := this.MustGet(setId)
	if err != nil {
		return nil, err
	}

	return getEntryDao(set), nil
}

func (this *_EntrySetCache) MustGet(setId string) (*model.Set, error) {
	if set, ok := this.cache[setId]; ok {
		return set, nil
	}

	set, err := SetDao.Find(nil, setId)
	if err != nil {
		return nil, err
	}

	this.MustSet(set)
	return set, nil
}

func (this *_EntrySetCache) MustSet(set *model.Set) {
	this.Lock()
	defer this.Unlock()

	if len(this.cache) > entrySetCacheLimit {
		newCache := make(map[string]*model.Set, entrySetCacheLimit)
		newCache[set.SetId] = set
		this.cache = newCache
	} else {
		this.cache[set.SetId] = set
	}
}

func getEntryDao(set *model.Set) EntryDAO {
	return getEntryDaoWithSourceAndJobType(set.SourceType, set.Type)
}

func getEntryDaoWithSourceAndJobType(sourceType enums.SourceType, jobType enums.JobType) EntryDAO {
	switch {
	case sourceType == enums.SourceTypeKodo && jobType == enums.JobTypeStream:
		return QnIncEntriesDao
	case sourceType == enums.SourceTypeKodo && jobType == enums.JobTypeBatch:
		return QnInvEntriesDao
	case sourceType == enums.SourceTypeApi && jobType == enums.JobTypeStream:
		return ApiIncEntriesDao
	case sourceType == enums.SourceTypeApi && jobType == enums.JobTypeBatch:
		return ApiInvEntriesDao
	default:
		return nil
	}
}
