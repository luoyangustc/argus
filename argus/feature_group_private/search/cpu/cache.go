package cpu

import (
	"qiniu.com/argus/feature_group_private/search"
)

type Cache struct {
	AllBlocks   []*Block
	BlockSize   int
	FeatureSets map[string]FeatureSet
}

func (c *Cache) AvaiableBlocks() []*Block {
	var emptyBlocks []*Block
	for _, block := range c.AllBlocks {
		if len(block.Owner) == 0 {
			emptyBlocks = append(emptyBlocks, block)
		}
	}
	return emptyBlocks
}

func (c *Cache) GetEmptyBlock(blockNum int) ([]*Block, error) {
	var emptyBlocks = c.AvaiableBlocks()
	if len(emptyBlocks) < blockNum {
		return nil, search.ErrNotEnoughBlocks
	}
	return emptyBlocks[:blockNum], nil
}

func (c *Cache) GetFeatureSets() map[string]FeatureSet {
	return c.FeatureSets
}

func NewCache(blockNum, blockSize int) (cache *Cache, err error) {
	cache = &Cache{
		BlockSize:   blockSize,
		FeatureSets: make(map[string]FeatureSet, 0),
	}
	buffer := make([]byte, blockNum*blockSize, blockNum*blockSize)
	for i := 0; i < blockNum; i++ {
		block := NewBlock(i, blockSize, buffer[i*blockSize:(i+1)*blockSize])
		cache.AllBlocks = append(cache.AllBlocks, block)
	}
	return
}

func (c *Cache) AccquireBuffer(size int) (buffer []byte, err error) {
	buffer = make([]byte, size)
	return
}
