package records

import (
	"container/list"
	"context"
)

var _ Records = &MockRecords{}

type MockRecords struct {
	m map[RecordKey]bool
	*list.List
}

func NewMock() *MockRecords {
	return &MockRecords{
		m:    make(map[RecordKey]bool),
		List: list.New(),
	}
}

func (mock *MockRecords) Count(context.Context) (int, error) {
	return mock.List.Len(), nil
}

func (mock *MockRecords) HasKey(ctx context.Context, key RecordKey) (bool, error) {
	return mock.m[key], nil
}

func (mock *MockRecords) Append(ctx context.Context,
	key RecordKey, value RecordValue,
) error {
	mock.List.PushBack(struct {
		Key   RecordKey
		Value RecordValue
	}{
		Key:   key,
		Value: value,
	})
	mock.m[key] = true
	return nil
}

func (mock *MockRecords) Close(context.Context) ([]string, error) { return nil, nil }
