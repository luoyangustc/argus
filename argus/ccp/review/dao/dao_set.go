package dao

import (
	"context"

	mgoutil "github.com/qiniu/db/mgoutil.v3"
	"gopkg.in/mgo.v2/bson"
	"qiniu.com/argus/ccp/review/enums"
	"qiniu.com/argus/ccp/review/model"
)

type _SetDAO interface {
	Find(ctx context.Context, setId string) (*model.Set, error)
	Insert(ctx context.Context, set *model.Set) error
	Query(ctx context.Context, query *SetFilter) ([]*model.Set, error)
	QueryBySets(context.Context, []string) ([]*model.Set, error)
	Remove(ctx context.Context, uid uint32, setId string) error
}

type SetDAOInMgo struct {
	c *mgoutil.Collection
}

func NewSetDAOInMgo(c *mgoutil.Collection) _SetDAO {
	return &SetDAOInMgo{c: c}
}

func (this *SetDAOInMgo) Find(ctx context.Context, setId string) (item *model.Set, err error) {
	query(this.c, func(c *mgoutil.Collection) {
		err = c.Find(bson.M{
			"set_id": setId,
		}).One(&item)
	})

	return
}

func (this *SetDAOInMgo) Insert(ctx context.Context, set *model.Set) (err error) {
	set.ID = bson.NewObjectId() // generate new object Id

	query(this.c, func(c *mgoutil.Collection) {
		err = c.Insert(set)
	})

	return
}

func (this *SetDAOInMgo) Query(ctx context.Context, filter *SetFilter) (items []*model.Set, err error) {
	query(this.c, func(c *mgoutil.Collection) {
		q := bson.M{
			"uid":         filter.Uid,
			"source_type": filter.SourceType,
			"type":        filter.Type,
		}

		if filter.Bucket != "" {
			q["bucket"] = filter.Bucket
		}

		if filter.Prefix != "" {
			q["prefix"] = filter.Prefix
		}

		// check 机审或机+人审
		if filter.Automatic || filter.Manual {
			q["automatic"] = filter.Automatic
			q["manual"] = filter.Manual
		}

		err = c.Find(q).Sort("-_id").All(&items)
	})

	return
}

func (this *SetDAOInMgo) QueryBySets(ctx context.Context, sets []string) (items []*model.Set, err error) {
	query(this.c, func(c *mgoutil.Collection) {
		q := bson.M{
			"set_id": bson.M{
				"$in": sets,
			},
		}

		err = c.Find(q).All(&items)
	})

	return
}

func (this *SetDAOInMgo) Remove(ctx context.Context, uid uint32, setId string) (err error) {
	query(this.c, func(c *mgoutil.Collection) {
		err = c.Remove(bson.M{
			"uid":    uid,
			"set_id": setId,
		})
	})
	return
}

type SetFilter struct {
	SetId string `json:"set_id"`

	Uid uint32 `json:"uid"`

	SourceType enums.SourceType `json:"source_type"`
	Type       enums.JobType    `json:"type"`

	Automatic bool `json:"automatic"`
	Manual    bool `json:"manual"`

	Bucket string `json:"bucket"`
	Prefix string `json:"prefix"`
}

func (this *SetFilter) IsValid() bool {
	if this.SetId != "" {
		return true
	}

	return this.SourceType.IsValid() && this.Type.IsValid() && this.Uid != 0
}

func (this *SetFilter) GetEntryDao() (EntryDAO, error) {
	if this.SetId != "" {
		return EntrySetCache.GetDao(this.SetId)
	}

	return getEntryDaoWithSourceAndJobType(this.SourceType, this.Type), nil
}

func (this *SetFilter) GetSetIds() ([]string, error) {
	if this.SetId != "" {
		return []string{this.SetId}, nil
	}

	sets, err := SetDao.Query(nil, this)
	if err != nil {
		return nil, err
	}

	var ret []string

	// if batch job should get last setId
	if this.Type == enums.JobTypeBatch && this.SourceType == enums.SourceTypeKodo {
		setIds := make(map[string]bool)

		ret = make([]string, 0)
		for _, set := range sets {
			if setIds[set.ResourceId()] {
				continue
			}
			setIds[set.ResourceId()] = true

			ret = append(ret, set.SetId)
		}
	} else {
		ret = make([]string, len(sets))
		for i, set := range sets {
			ret[i] = set.SetId
		}
	}

	return ret, nil
}
