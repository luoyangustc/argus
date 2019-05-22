package dao

const (
	defaultPaginatorLimit = 20
)

type Paginator struct {
	Offset int
	Limit  int
}

func NewPaginator(offset, limit int) *Paginator {
	return &Paginator{
		Offset: offset,
		Limit:  limit,
	}
}

func (this *Paginator) IsValid() bool {
	return this.Offset >= 0 && this.Limit > 0
}
