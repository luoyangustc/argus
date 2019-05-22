package inference

import (
	"bytes"
	"context"
	"image"
	"io"
	"os"

	_ "image/jpeg"
	_ "image/png"

	"qiniu.com/argus/atserving/model"
)

type Native struct{}

func NewNative() Creator { return Native{} }

func (native Native) Create(ctx context.Context, params *CreateParams) (Instance, error) {

	switch params.App {
	case "image":
		return ImageInstance{}, nil
	}

	return nil, nil
}

////////////////////////////////////////////////////////////////////////////////

type ImageInstance struct{}

func (ii ImageInstance) Preprocess(
	ctx context.Context, req model.EvalRequestInner,
) (model.EvalRequestInner, error) {
	return req, nil
}

func (ii ImageInstance) PreprocessGroup(
	ctx context.Context, req model.GroupEvalRequestInner,
) (model.GroupEvalRequestInner, error) {
	return req, nil
}

func (ii ImageInstance) Inference(
	ctx context.Context, reqs []model.EvalRequestInner,
) ([]Response, error) {

	resps := make([]Response, len(reqs))

	for i, req := range reqs {

		info, err := func(uri interface{}) (interface{}, error) {
			var r io.Reader
			switch v := uri.(type) {
			case []byte:
				r = bytes.NewReader(v)
			case model.STRING:
				f, err := os.Open(v.String())
				if err != nil {
					return nil, err
				}
				defer f.Close()
			}

			_image, format, err := image.Decode(r)
			if err != nil {
				return nil, err
			}
			return struct {
				Format string `json:"format"`
				Width  int    `json:"width"`
				Height int    `json:"height"`
			}{
				Format: format,
				Width:  _image.Bounds().Dx(),
				Height: _image.Bounds().Dy(),
			}, nil
		}(req.Data.URI)

		if err != nil {
			resps[i].Code = 500
			resps[i].Message = err.Error()
		} else {
			resps[i].Result = struct {
				Metadata interface{} `json:"metadata"`
			}{
				Metadata: info,
			}
		}
	}

	return resps, nil
}

func (ii ImageInstance) InferenceGroup(
	ctx context.Context, reqs []model.GroupEvalRequestInner,
) ([]Response, error) {
	return nil, nil
}
