package dao

import "gopkg.in/mgo.v2/bson"

const (
	defaultPaginatorLimit = 20
)

type Paginator struct {
	Marker string `json:"marker"`
	Limit  int    `json:"limit"`
}

func NewPaginator(marker string, limit int) *Paginator {
	if limit == 0 {
		limit = defaultPaginatorLimit
	}

	return &Paginator{
		Marker: marker,
		Limit:  limit,
	}
}

func (this *Paginator) IsValid() bool {
	return this.Marker != "" && bson.IsObjectIdHex(this.Marker)
}
