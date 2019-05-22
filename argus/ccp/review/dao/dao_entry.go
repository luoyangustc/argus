package dao

import (
	"context"
	"fmt"
	"time"

	mgoutil "github.com/qiniu/db/mgoutil.v3"
	"gopkg.in/mgo.v2/bson"
	"qiniu.com/argus/ccp/review/enums"
	"qiniu.com/argus/ccp/review/model"
)

type EntryDAO interface {
	Query(context.Context, *EntryFilter, *Paginator) ([]*model.Entry, error)
	Find(context.Context, string) (*model.Entry, error)
	Insert(context.Context, *model.Entry) error
	BatchInsert(context.Context, []*model.Entry) error
	Update(context.Context, string, *model.Entry) error
	Count(context.Context, string, enums.Suggestion) (int, error)
	Remove(ctx context.Context, setId string) error
}

type EntryDAOInMgo struct {
	c *mgoutil.Collection
}

func NewEntryDAOInMgo(c *mgoutil.Collection) EntryDAO {
	return &EntryDAOInMgo{c: c}
}

type EntryFilter struct {
	SetIds []string

	Mimetype   enums.MimeType   `json:"mimetype"`
	Suggestion enums.Suggestion `json:"suggestion"`

	Scene enums.Scene `json:"scene"`
	Min   float32     `json:"min"`
	Max   float32     `json:"max"`

	StartAt int64 `json:"start"`
	EndAt   int64 `json:"end"`
}

func (this *EntryFilter) toQueryParams() bson.M {
	q := bson.M{
		"set_id": bson.M{
			"$in": this.SetIds,
		},
	}

	if this.Mimetype.IsValid() {
		q["mimetype"] = this.Mimetype
	}

	if this.Scene.IsValid() {
		if this.Suggestion.IsValid() {
			q["original.suggestion"] = this.Suggestion

			// for scene suggestion
			ssKey := fmt.Sprintf("original.scenes.%s.suggestion", this.Scene)
			if this.Suggestion == enums.SuggestionDisabled {
				q[ssKey] = enums.SuggestionBlock
			} else {
				q[ssKey] = this.Suggestion
			}
		}

		if this.Min > 0 && this.Max > this.Min {
			q[fmt.Sprintf("original.scenes.%s.score", this.Scene)] = bson.M{
				"$gt": this.Min,
				"$lt": this.Max,
			}
		}

		q["final"] = nil
	} else if this.Suggestion.IsValid() {
		q["original.suggestion"] = this.Suggestion
		// hack way for all unreview data.
		q["final"] = nil
	}

	if this.StartAt != 0 && this.EndAt != 0 {
		q["created_at"] = bson.M{
			"$gt": this.StartAt,
			"$lt": this.EndAt,
		}
	}

	return q
}

func (this *EntryDAOInMgo) Query(ctx context.Context, filter *EntryFilter, p *Paginator) (items []*model.Entry, err error) {
	items = make([]*model.Entry, 0)

	query(this.c, func(c *mgoutil.Collection) {
		q, limit := getQueryParamsWithPaginator(filter.toQueryParams(), p, false)
		err = c.Find(q).Sort("-_id").Limit(limit).All(&items)
	})

	return
}

func (this *EntryDAOInMgo) Find(ctx context.Context, id string) (item *model.Entry, err error) {
	if !bson.IsObjectIdHex(id) {
		return nil, ErrInvalidId
	}

	query(this.c, func(c *mgoutil.Collection) {
		err = c.FindId(bson.ObjectIdHex(id)).One(&item)
	})

	return
}

func (this *EntryDAOInMgo) Insert(ctx context.Context, entry *model.Entry) (err error) {
	entry.ID = bson.NewObjectId() // generate new ID
	entry.CreatedAt = time.Now().Unix()
	entry.Final = nil

	cuts := entry.GetVideoCuts()

	query(this.c, func(c *mgoutil.Collection) {
		if err = c.Insert(entry); err == nil && cuts != nil {
			_ = VideoCutDAO.BatchInsert(ctx, cuts)
		}
	})

	return
}

func (this *EntryDAOInMgo) BatchInsert(ctx context.Context, entries []*model.Entry) (err error) {
	docs := make([]interface{}, len(entries))
	unixTime := time.Now().Unix()

	var cuts []*model.VideoCut

	for i, entry := range entries {
		entry.ID = bson.NewObjectId()
		entry.CreatedAt = unixTime
		entry.Final = nil
		docs[i] = entry

		// auto save video cuts
		if vCuts := entry.GetVideoCuts(); vCuts != nil {
			cuts = append(cuts, vCuts...)
		}
	}

	query(this.c, func(c *mgoutil.Collection) {
		if err = c.Insert(docs...); err == nil && cuts != nil {
			_ = VideoCutDAO.BatchInsert(ctx, cuts)
		}
	})
	return
}

func (this *EntryDAOInMgo) Update(ctx context.Context, id string, entry *model.Entry) (err error) {
	if !bson.IsObjectIdHex(id) {
		return ErrInvalidId
	}

	query(this.c, func(c *mgoutil.Collection) {
		err = c.Update(bson.M{
			"_id": bson.ObjectIdHex(id),
		}, entry)
	})

	return
}

func (this *EntryDAOInMgo) Count(ctx context.Context, setId string, suggestion enums.Suggestion) (n int, err error) {
	query(this.c, func(c *mgoutil.Collection) {
		n, err = c.Find(bson.M{
			"set_id":              setId,
			"original.suggestion": suggestion,
		}).Count()
	})

	return
}

func (this *EntryDAOInMgo) Remove(ctx context.Context, setId string) (err error) {
	query(this.c, func(c *mgoutil.Collection) {
		_, err = c.RemoveAll(bson.M{
			"set_id": setId,
		})
	})
	return
}
