// +build !zmq

package inference

import (
	"context"

	"github.com/qiniu/xlog.v1"
)

type zmqInstanceer struct {
}

func NewZmq() Creator { return zmqInstanceer{} }

func (zmqI zmqInstanceer) Create(ctx context.Context, params *CreateParams) (Instance, error) {
	xl := xlog.FromContextSafe(ctx)
	xl.Panicln("compile no ZMQ tag")
	return nil, nil
}
