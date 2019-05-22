package dao

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"gopkg.in/mgo.v2/bson"
)

func TestPaginator(t *testing.T) {
	assertion := assert.New(t)

	defaultPaginator := NewPaginator("", 0)
	assertion.Equal(defaultPaginator.Limit, defaultPaginatorLimit)
	assertion.False(defaultPaginator.IsValid())

	validPaginator := NewPaginator(bson.NewObjectId().Hex(), 25)
	assertion.Equal(25, validPaginator.Limit)
	assertion.True(validPaginator.IsValid())
}
