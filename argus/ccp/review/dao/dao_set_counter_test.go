package dao

import (
	"context"
	"testing"

	"qiniu.com/argus/ccp/review/enums"

	"github.com/stretchr/testify/assert"
	"gopkg.in/mgo.v2/bson"
	"qiniu.com/argus/ccp/review/model"
)

func TestSetCounterDAOInit(t *testing.T) {
	assertion := assert.New(t)
	assertion.NotNil(SetCounterDAO)
}

func TestSetCounterInsert(t *testing.T) {
	assertion := assert.New(t)

	counter := &model.SetCounter{
		UserId:     1,
		SetId:      bson.NewObjectId().Hex(),
		ResourceId: "resource_id",
		Values: map[enums.Scene]int{
			enums.ScenePulp:       2050,
			enums.SceneTerror:     2050,
			enums.ScenePolitician: 2050,
		},
	}

	err := SetCounterDAO.Insert(context.Background(), counter)
	assertion.Nil(err)

	foundCounter, err := SetCounterDAO.Find(context.Background(), counter.SetId)
	assertion.Nil(err)
	assertion.NotNil(foundCounter)

	assertion.Equal(foundCounter.ID.Hex(), counter.ID.Hex())
	assertion.Equal(foundCounter.ResourceId, counter.ResourceId)
	assertion.Equal(foundCounter.Values[enums.ScenePulp], counter.Values[enums.ScenePulp])
}

func TestSetCounterUpdate(t *testing.T) {
	assertion := assert.New(t)

	counter := &model.SetCounter{
		UserId:     1,
		SetId:      bson.NewObjectId().Hex(),
		ResourceId: "resource_id",
		Values: map[enums.Scene]int{
			enums.ScenePulp:       2050,
			enums.SceneTerror:     2050,
			enums.ScenePolitician: 2050,
		},
	}

	counter.LelfValues = counter.Values

	err := SetCounterDAO.Insert(nil, counter)
	assertion.Nil(err)

	foundCounter, err := SetCounterDAO.Find(nil, counter.SetId)
	assertion.Nil(err)
	assertion.NotNil(foundCounter)

	foundCounter.Values[enums.ScenePulp] = 3000
	foundCounter.LelfValues[enums.ScenePulp] = 25

	_ = SetCounterDAO.Update(nil, foundCounter, 0)

	foundCounter, err = SetCounterDAO.Find(nil, counter.SetId)
	assertion.Nil(err)
	assertion.NotNil(foundCounter)

	assertion.Equal(foundCounter.Values[enums.ScenePulp], 3000)
	assertion.Equal(foundCounter.LelfValues[enums.ScenePulp], 25)
}

func TestSetCounterQuery(t *testing.T) {
	assertion := assert.New(t)

	counter := &model.SetCounter{
		UserId:     1,
		SetId:      bson.NewObjectId().Hex(),
		ResourceId: "resource_id",
		Values: map[enums.Scene]int{
			enums.ScenePulp:       2050,
			enums.SceneTerror:     2050,
			enums.ScenePolitician: 2050,
		},
	}

	counter.LelfValues = counter.Values

	err := SetCounterDAO.Insert(nil, counter)
	assertion.Nil(err)

	// items, err := SetCounterDAO.Query(nil, counter.UserId, []string{
	// 	counter.SetId,
	// }, nil)

	// assertion.Nil(err)
	// assertion.NotEmpty(items)
}

func TestSetCounter_Remove(t *testing.T) {
	assertion := assert.New(t)

	{
		setId := bson.NewObjectId().Hex()
		counter := &model.SetCounter{
			UserId:     1,
			SetId:      setId,
			ResourceId: "resource_id",
			Values: map[enums.Scene]int{
				enums.ScenePulp:       2050,
				enums.SceneTerror:     2050,
				enums.ScenePolitician: 2050,
			},
		}
		err := SetCounterDAO.Insert(context.Background(), counter)
		assertion.Nil(err)
		err = SetCounterDAO.Remove(context.Background(), 1, setId)
		assertion.Nil(err)
	}
}
