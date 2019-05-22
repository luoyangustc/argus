package uri

import (
	"bytes"
	"context"
	"encoding/base64"
	"io/ioutil"
	"strings"
)

type dataHandler struct {
}

func (h *dataHandler) Get(ctx context.Context, args Request, opts ...GetOption,
) (resp *Response, err error) {

	data := strings.TrimPrefix(args.URI, "data:application/octet-stream;base64,")
	bs, err := base64.StdEncoding.DecodeString(data)
	if err != nil {
		return nil, err
	}
	return &Response{
		Body: ioutil.NopCloser(bytes.NewBuffer(bs)),
		Size: int64(len(bs)),
	}, nil
}

func (h *dataHandler) Names() []string {
	return []string{"data"}
}

func WithDataHandler() Handler {
	return &dataHandler{}
}
