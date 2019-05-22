package fetcher

import (
	"context"
	"net/http"

	"qbox.us/net/httputil"
	authstub "qiniu.com/auth/authstub.v1"
)

type Server interface {
	PostVideoConfig(ctx context.Context, req *VideoConfigReq, env *authstub.Env)
	PostBlacklist(ctx context.Context, req *BlacklistReq, env *authstub.Env)
	GetMetrics(ctx context.Context, req *BaseReq, env *authstub.Env)
}

var _ Server = &server{}

type server struct {
	kodo Kodo
}

func NewServer(kodo Kodo) Server {
	return &server{
		kodo: kodo,
	}
}

type VideoConfigReq struct {
	CmdArgs      []string
	MaxVideoSize int `json:"max_video_size"`
}

func (s *server) PostVideoConfig(ctx context.Context, req *VideoConfigReq, env *authstub.Env) {
	// 设置短视频最大文件大小
	if req.MaxVideoSize > 0 {
		s.kodo.SetMaxVideoSize(req.MaxVideoSize)
	}
}

type BlacklistReq struct {
	CmdArgs []string
	Lists   []struct {
		UID     uint32   `json:"uid"`
		Buckets []string `json:"buckets"`
	} `json:"lists"`
}

func (s *server) PostBlacklist(ctx context.Context, req *BlacklistReq, env *authstub.Env) {
	for _, bl := range req.Lists {
		if err := s.kodo.AddBlackList(ctx, bl.UID, bl.Buckets); err != nil {
			httputil.ReplyErr(env.W, http.StatusBadRequest, "fail to add blacklist")
			return
		}
	}
}

type BaseReq struct {
	CmdArgs []string
}

func (s *server) GetMetrics(ctx context.Context, req *BaseReq, env *authstub.Env) {
	httputil.Reply(env.W, http.StatusOK, s.kodo.GetMetrics())
}
