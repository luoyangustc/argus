package main

import (
	"context"

	"github.com/qiniu/http/restrpc.v1"
)

type Service struct {
	*Run
}

func (s Service) PostConfig(
	ctx context.Context,
	req *struct {
		PoolSize int `json:"pool_size"`
	},
	env *restrpc.Env,
) (interface{}, error) {

	if req.PoolSize > 0 {
		if req.PoolSize < s.Run.Config.PoolSize {
			for i := 0; i < s.Run.Config.PoolSize-req.PoolSize; i++ {
				s.handles <- true
			}
			s.Run.Config.PoolSize = req.PoolSize
		} else if req.PoolSize > s.Run.Config.PoolSize {
			for i := 0; i < req.PoolSize-s.Run.Config.PoolSize; i++ {
				<-s.handles
			}
			s.Run.Config.PoolSize = req.PoolSize
		}
	}

	return req, nil
}

type Callback struct {
	ID      string      `json:"id"`
	OP      string      `json:"op"`
	Code    int         `json:"code"`
	Message string      `json:"message"`
	Result  interface{} `json:"result"`
}

func (s Service) PostCallback(
	ctx context.Context,
	req *Callback,
	env *restrpc.Env,
) error {
	s.callbacks <- *req
	return nil
}
