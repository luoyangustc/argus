package hub

import (
	"context"
	"testing"

	"qiniu.com/argus/tuso/proto"
	"qiniu.com/auth/authstub.v1"
	authProto "qiniu.com/auth/proto.v1"

	"github.com/pkg/errors"
	"github.com/stretchr/testify/assert"
)

func TestServer(t *testing.T) {
	env := prepareTestEnv(t)
	s := env.s
	ctx := context.Background()
	uinfo := &authstub.Env{UserInfo: authProto.UserInfo{Uid: 111}}
	t.Run("add hub", func(t *testing.T) {
		a := assert.New(t)
		err := s.PostHub(ctx, &proto.PostHubReq{Name: "firsthub", Bucket: "test", Prefix: ""}, uinfo)
		a.Nil(err)
	})
	t.Run("add hub repeat", func(t *testing.T) {
		a := assert.New(t)
		err := s.PostHub(ctx, &proto.PostHubReq{Name: "firsthub", Bucket: "test", Prefix: ""}, uinfo)
		a.Equal(ErrHubExists, errors.Cause(err))
	})
	t.Run("add hub, bad name", func(t *testing.T) {
		a := assert.New(t)
		err := s.PostHub(ctx, &proto.PostHubReq{Name: "0000", Bucket: "test", Prefix: ""}, uinfo)
		a.NotNil(err)
	})
	t.Run("add image, hub not exists", func(t *testing.T) {
		a := assert.New(t)
		resp, err := s.PostImage(ctx, &proto.PostImageReq{
			Hub: "xxx",
			Op:  "ADD",
		}, uinfo)
		a.Equal(ErrHubNotFound, errors.Cause(err))
		a.Nil(resp)
	})
	t.Run("add image, hub exists, but uid mismatches", func(t *testing.T) {
		a := assert.New(t)
		resp, err := s.PostImage(ctx, &proto.PostImageReq{
			Hub: "firsthub",
			Op:  "ADD",
		}, &authstub.Env{UserInfo: authProto.UserInfo{Uid: 222}})
		a.Equal(ErrHubNotFound, errors.Cause(err))
		a.Nil(resp)
	})
	t.Run("add image, success", func(t *testing.T) {
		a := assert.New(t)
		resp, err := s.PostImage(ctx, &proto.PostImageReq{
			Hub: "firsthub",
			Op:  "ADD",
			Images: []proto.ImageKey{
				proto.ImageKey{
					Key: "1.jpg",
				},
				proto.ImageKey{
					Key: "2.jpg",
				},
			},
		}, uinfo)
		a.Nil(err)
		a.Equal(2, resp.SuccessCnt)
		n, err := s.db.OpLog.Count()
		a.Nil(err)
		a.Equal(2, n)
	})
	t.Run("add image, success, repeat", func(t *testing.T) {
		a := assert.New(t)
		resp, err := s.PostImage(ctx, &proto.PostImageReq{
			Hub: "firsthub",
			Op:  "ADD",
			Images: []proto.ImageKey{
				proto.ImageKey{
					Key: "1.jpg",
				},
				proto.ImageKey{
					Key: "2.jpg",
				},
			},
		}, uinfo)
		a.Nil(err)
		a.Equal(0, resp.SuccessCnt)
		n, err := s.db.OpLog.Count()
		a.Nil(err)
		a.Equal(4, n)
	})
	// TODO: update image
	// TODO: remove image
}
