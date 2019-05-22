package pool

import (
	"errors"
)

var (
	EOF = errors.New("eof")
)

// ----------------------------------------------------------

const (
	N = 1024
)

type Pool struct {
	pages [][]interface{}
	head  *interface{}
}

func (p *Pool) newPage() {

	page := make([]interface{}, N)
	for i := 1; i < N; i++ {
		page[i-1] = &page[i]
	}
	page[N-1] = p.head

	p.pages = append(p.pages, page)
	p.head = &page[0]
}

func (p *Pool) Add(v interface{}) *interface{} {

	if p.head == nil {
		p.newPage()
	}
	addr := p.head
	p.head = (*addr).(*interface{})
	*addr = v
	return addr
}

func (p *Pool) Free(addr *interface{}) {

	*addr = p.head
	p.head = addr
}

func (p *Pool) ForPage(ipage int, doWhat func(v interface{})) error {

	if ipage >= len(p.pages) {
		return EOF
	}

	for _, v := range p.pages[ipage] {
		if _, ok := v.(*interface{}); !ok {
			doWhat(v)
		}
	}
	return nil
}

func (p *Pool) ForEach(doWhat func(v interface{})) {

	for _, page := range p.pages {
		for _, v := range page {
			if _, ok := v.(*interface{}); !ok {
				doWhat(v)
			}
		}
	}
}

// ----------------------------------------------------------
