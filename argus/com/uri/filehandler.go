package uri

import (
	"context"
	"os"
	"strings"

	"github.com/pkg/errors"
)

type fileHandler struct{}

func (h *fileHandler) Get(ctx context.Context, args Request, opts ...GetOption,
) (resp *Response, err error) {

	fileName := strings.TrimPrefix(args.URI, "file://")
	f, err := os.Open(fileName)
	if err != nil {
		return nil, errors.Wrapf(err, "os.Open %q", fileName)
	}
	fi, err := f.Stat()
	if err != nil {
		return nil, errors.Wrapf(err, "f.Stat %q", fileName)
	}
	n := fi.Size()
	resp = &Response{
		Size: n,
		Body: f,
	}
	return resp, nil
}

func (h *fileHandler) Names() []string {
	return []string{"file"}
}

func WithFileHandler() Handler {
	return &fileHandler{}
}
