package video

import (
	"context"
	"fmt"
	"testing"

	"qiniu.com/argus/video/vframe"
	authstub "qiniu.com/auth/authstub.v1"
)

type MockOPs struct {
}

func (mock MockOPs) ResetOP(op string, config *OPConfig) {}
func (mock MockOPs) Create(ctx context.Context, params map[string]OPParams, env OPEnv) (map[string]OP, bool) {
	_ops := make(map[string]OP)
	for key, _ := range params {
		_ops[key] = MockOP{
			eval: func(uris []string) []interface{} {
				return make([]interface{}, len(uris))
			},
		}
	}
	return _ops, true
}

type MockOP struct {
	eval func(uris []string) []interface{}
}

func (mock MockOP) Fork(ctx context.Context, params OPParams, env OPEnv) (OP, bool) {
	return mock, true
}

func (mock MockOP) Reset(context.Context) error { return nil }
func (mock MockOP) Count() int32                { return MAX_OP_COUNT }

func (mock MockOP) Eval(ctx context.Context, uris []string) ([]interface{}, error) {
	return mock.eval(uris), nil
}

func (mock MockOP) NewCuts(
	ctx context.Context, params, originParams *vframe.VframeParams,
) CutsV2 {
	return NewSimpleCutsV2(params, originParams,
		func(ctx context.Context, uris []string) ([]interface{}, error) {
			return mock.Eval(ctx, uris)
		},
		func(ctx context.Context, _v interface{}) ([]string, []float32, error) {
			return []string{"x"}, []float32{1.0}, nil
		},
	)
}

func (mock MockOP) NewMeta(ctx context.Context, params SegmentParams) OPMeta {
	return NewSimpleCutOPMeta(params, OPParams{}, nil, nil, true)
}

type MockEndHook struct {
}

func (mock MockEndHook) End(ctx context.Context, result EndResult) error {
	fmt.Printf("end: %#v\n", result)
	return nil
}

func TestVideo(t *testing.T) {

	ch := make(chan vframe.CutResponse)
	go func() {
		for i := 0; i < 20; i++ {
			{
				resp := vframe.CutResponse{}
				resp.Result.Cut =
					struct {
						Offset int64  `json:"offset"`
						URI    string `json:"uri"`
					}{
						Offset: int64(i),
						URI:    "file:///a",
					}
				ch <- resp
			}
		}
		close(ch)
	}()

	d := video{
		Vframe: vframe.MockVframe{
			NewJob: func(req vframe.VframeRequest) vframe.Job {
				return vframe.MockJob{CH: ch}
			},
		},
	}
	// d.createEndHook = func(url string, op string) EndHook {
	// 	return MockEndHook{}
	// }

	req := VideoRequest{}
	req.Data.URI = "file:///xx.mp4"
	req.Ops = append(req.Ops,
		struct {
			OP             string   `json:"op"`
			CutHookURL     string   `json:"cut_hook_url"`
			SegmentHookURL string   `json:"segment_hook_url"`
			HookURL        string   `json:"hookURL"`
			Params         OPParams `json:"params"`
		}{
			OP: "aa",
		},
		struct {
			OP             string   `json:"op"`
			CutHookURL     string   `json:"cut_hook_url"`
			SegmentHookURL string   `json:"segment_hook_url"`
			HookURL        string   `json:"hookURL"`
			Params         OPParams `json:"params"`
		}{
			OP: "bb",
		},
	)
	var env authstub.Env
	env.UserInfo.Uid = 111
	env.UserInfo.Utype = 111
	ops, _ := MockOPs{}.Create(context.Background(),
		map[string]OPParams{},
		OPEnv{Uid: env.UserInfo.Uid, Utype: env.UserInfo.Utype},
	)
	d.Run(context.Background(), req, ops, nil,
		func(op string) EndHook {
			return MockEndHook{}
		},
		nil, nil)

}
