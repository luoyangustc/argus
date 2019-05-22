package bucket_scan

import (
	"container/list"
	"context"

	"qiniu.com/argus/argus/com/bucket"
)

type mockScanner struct {
	*list.List
}

func NewMockScanner(lst *list.List) bucket.IScanner {
	return mockScanner{List: lst}
}

func (mock mockScanner) Scan(context.Context, int) (bucket.IIter, error) {
	return &mockIter{
		List:    mock.List,
		Element: mock.List.Front(),
	}, nil
}
func (mock mockScanner) Count(context.Context) (int, error) {
	return mock.List.Len(), nil
}

type mockIter struct {
	*list.List
	*list.Element
}

func (mock *mockIter) Next(context.Context) (bucket.KeyItem, string, bool) {
	if mock.Element == nil {
		return bucket.KeyItem{}, "", false
	}
	var e = mock.Element
	mock.Element = mock.Element.Next()
	return e.Value.(bucket.KeyItem), "", true
}
func (mock mockIter) Error() error { return nil }
