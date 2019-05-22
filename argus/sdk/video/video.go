package video

import "context"

func Video(ctx context.Context,
	vframe Vframe, pipe CutsPipe,
	cutHook func(context.Context, CutResponse)) error {

	bufsize := 16
	for {
		cuts, ok := vframe.Next(ctx, bufsize)
		if !ok {
			break
		}
		for _, resp := range pipe.Append(ctx, cuts...) {
			cutHook(ctx, resp)
		}
	}
	if err := vframe.Error(); err != nil {
		return err
	}
	for _, resp := range pipe.End(ctx) {
		cutHook(ctx, resp)
	}

	return nil
}

func VideoEnd(ctx context.Context,
	vframe Vframe, pipe CutsPipe) ([]CutResponse, error) {
	resps := make([]CutResponse, 0)
	err := Video(ctx, vframe, pipe, func(_ context.Context, resp CutResponse) {
		resps = append(resps, resp)
	})
	return resps, err
}

func Images(ctx context.Context,
	images [][]byte, pipe CutsPipe) ([]CutResponse, error) {
	reqs := make([]CutRequest, 0, len(images))
	for i, image := range images {
		reqs = append(reqs, CutRequest{OffsetMS: int64(i), Body: image})
	}
	return pipe.Append(ctx, reqs...), nil
}
