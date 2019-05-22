package hub

import (
	"context"
	"fmt"
	"testing"

	"qiniu.com/argus/tuso/proto"
	"qiniu.com/auth/authstub.v1"
	authProto "qiniu.com/auth/proto.v1"

	"github.com/stretchr/testify/assert"
)

func TestInternalServer(t *testing.T) {
	defer setUpTestEnvLog(t)()
	env := prepareTestEnv(t)
	s := env.s
	o := env.o
	ins := env.ins
	ctx := context.Background()
	uinfo := &authstub.Env{UserInfo: authProto.UserInfo{Uid: 111}}
	t.Run("pre data", func(t *testing.T) {
		t.Run("add hub and image, success", func(t *testing.T) {
			a := assert.New(t)
			err := s.PostHub(ctx, &proto.PostHubReq{Name: "firsthub", Bucket: "test", Prefix: ""}, uinfo)
			a.Nil(err)

			for i := 0; i < proto.KodoBlockFeatureSize*2+40; i++ {
				resp, err := s.PostImage(ctx, &proto.PostImageReq{
					Hub: "firsthub",
					Op:  "ADD",
					Images: []proto.ImageKey{
						proto.ImageKey{
							Key: fmt.Sprintf("%d-1.jpg", i),
						},
						proto.ImageKey{
							Key: fmt.Sprintf("%d-2.jpg", i),
						},
						proto.ImageKey{
							Key: fmt.Sprintf("%d-3.jpg", i),
						},
					},
				}, uinfo)
				a.Equal(3, resp.SuccessCnt)
				a.Nil(err)
			}
		})
	})
	t.Run("get hub info, success", func(t *testing.T) {
		a := assert.New(t)
		resp, err := ins.GetHubInfo(ctx, &proto.GetHubInfoReq{
			HubName: "firsthub",
			Version: proto.DefaultFeatureVersion,
		}, nil)
		a.Nil(err)
		a.Equal("test", resp.Bucket)
	})
	t.Run("get hub info, hub not found", func(t *testing.T) {
		a := assert.New(t)
		_, err := ins.GetHubInfo(ctx, &proto.GetHubInfoReq{
			HubName: "firsxxxthub",
			Version: proto.DefaultFeatureVersion,
		}, nil)
		a.NotNil(err)
		a.Equal(ErrHubNotFound, err)
	})
	t.Run("get hub info, version not found", func(t *testing.T) {
		a := assert.New(t)
		_, err := ins.GetHubInfo(ctx, &proto.GetHubInfoReq{
			HubName: "firsthub",
			Version: proto.DefaultFeatureVersion + 1,
		}, nil)
		a.NotNil(err)
		a.Equal(ErrHubNotFound, err)
	})
	t.Run("get file meta info, success, default index and offset is -1", func(t *testing.T) {
		a := assert.New(t)
		resp, err := ins.GetFilemetaInfo(ctx, &proto.GetFileMetaInfoReq{
			HubName:           "firsthub",
			FeatureFileIndex:  -1,
			FeatureFileOffset: -1,
		}, nil)
		a.Nil(err)
		a.Equal("0-1.jpg", resp.Key)
		a.Equal("INIT", resp.Status)
	})
	t.Run("process and upload", func(t *testing.T) {
		a := assert.New(t)
		ctx, cancel := context.WithCancel(ctx)
		defer cancel()
		env.m.mode = ""
		_, err := o.processAllOpLog(ctx)
		a.Equal(nil, err)
		_, err = o.uploadKodo(ctx)
		a.Nil(err)
	})
	t.Run("get file meta info, success", func(t *testing.T) {
		a := assert.New(t)
		{
			resp, err := ins.GetFilemetaInfo(ctx, &proto.GetFileMetaInfoReq{
				HubName:           "firsthub",
				FeatureFileIndex:  0,
				FeatureFileOffset: 0,
			}, nil)
			a.Nil(err)
			a.Equal("0-1.jpg", resp.Key)
			a.Equal("OK", resp.Status)
		}
		{
			resp, err := ins.GetFilemetaInfo(ctx, &proto.GetFileMetaInfoReq{
				HubName:           "firsthub",
				FeatureFileIndex:  0,
				FeatureFileOffset: 1,
			}, nil)
			a.Nil(err)
			a.Equal("0-2.jpg", resp.Key)
			a.Equal("OK", resp.Status)
		}
	})
	t.Run("get file meta info, file not found", func(t *testing.T) {
		a := assert.New(t)
		{
			_, err := ins.GetFilemetaInfo(ctx, &proto.GetFileMetaInfoReq{
				HubName:           "firsthub",
				FeatureFileIndex:  8888,
				FeatureFileOffset: 0,
			}, nil)
			a.Equal(err, ErrFileNotFound)
		}
	})
}
