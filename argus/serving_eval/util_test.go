package eval

import (
	"context"
	"io"
	"net/http"

	STS "qiniu.com/argus/sts/client"
)

var _ STS.Client = &mockSTS{}

type mockSTS struct {
	newKey string
}

func newMockSTS() *mockSTS { return &mockSTS{} }
func (mock *mockSTS) NewURL(ctx context.Context, length *int64) (string, error) {
	return mock.newKey, nil
}
func (mock *mockSTS) GetURL(ctx context.Context, uri string, length *int64, options *STS.URIOptions) (string, error) {
	return uri, nil
}
func (mock *mockSTS) DoFetch(ctx context.Context, uri string, length *int64, sync bool) (string, int64, error) {
	return "", 0, nil
}
func (mock *mockSTS) Post(ctx context.Context, uri string, length int64, r io.Reader) error {
	return nil
}
func (mock *mockSTS) SyncPost(
	ctx context.Context,
	uri string, length int64, r io.Reader,
	syncDone func(err error),
) error {
	syncDone(nil)
	return nil
}
func (mock *mockSTS) Get(ctx context.Context, uri string, length *int64, options ...STS.GetOption) (io.ReadCloser, int64, http.Header, error) {
	return nil, 0, nil, nil
}
