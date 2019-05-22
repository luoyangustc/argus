package hub

import (
	"context"
	"fmt"
	"sync"
	"testing"

	"gopkg.in/mgo.v2/bson"

	"qiniu.com/argus/tuso/proto"
	"qiniu.com/auth/authstub.v1"
	authProto "qiniu.com/auth/proto.v1"

	"github.com/stretchr/testify/assert"
)

func TestOpLogPocess(t *testing.T) {
	defer setUpTestEnvLog(t)()
	for _, mode := range []int{0, 1} {
		env := prepareTestEnv(t)
		s := env.s
		o := env.o
		ctx := context.Background()

		uinfo := &authstub.Env{UserInfo: authProto.UserInfo{Uid: 111}}
		t.Run("prepare data, add hub, and add image", func(t *testing.T) {
			a := assert.New(t)
			err := s.PostHub(ctx, &proto.PostHubReq{Name: "firsthub", Bucket: "test", Prefix: ""}, uinfo)
			a.Nil(err)
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
			a.True(n == 2)

			var oplogs []dOpLog
			a.Nil(o.db.OpLog.Find(nil).All(&oplogs))
			a.Len(oplogs, 2)
			if len(oplogs) == 2 {
				a.Nil(oplogs[0].Feature)
				a.Nil(oplogs[1].Feature)
				a.Len(oplogs[0].Md5, 0)
				a.Len(oplogs[1].Md5, 0)
				a.Equal(OptatusInit, oplogs[0].Status)
				a.Equal(OptatusInit, oplogs[1].Status)
			}
		})
		t.Run("startProcessTask, cancel success", func(t *testing.T) {
			a := assert.New(t)
			ctx, cancel := context.WithCancel(ctx)
			cancel()
			w := sync.WaitGroup{}
			w.Add(1)
			go func() {
				err := o.processAllOpLogLoop(ctx)
				a.Equal(context.Canceled, err)
				w.Done()
			}()
			w.Wait()
		})
		if mode == 0 {
			t.Run("startProcessTask, process success", func(t *testing.T) {
				a := assert.New(t)
				ctx, cancel := context.WithCancel(ctx)
				defer cancel()
				w := sync.WaitGroup{}
				w.Add(1)
				env.m.mode = ""
				go func() {
					n, err := o.processAllOpLog(ctx)
					a.Equal(nil, err)
					a.Equal(2, n)
					w.Done()
				}()
				w.Wait()
				var oplogs []dOpLog
				a.Nil(o.db.OpLog.Find(nil).All(&oplogs))
				a.Len(oplogs, 2)
				if len(oplogs) == 2 {
					a.Len(oplogs[0].Feature, proto.FeatureSize)
					a.Len(oplogs[1].Feature, proto.FeatureSize)
					a.Len(oplogs[0].Md5, 32)
					a.Len(oplogs[1].Md5, 32)
					a.Equal(OptatusEvaled, oplogs[0].Status)
					a.Equal(OptatusEvaled, oplogs[1].Status)
				}
			})
		} else {
			t.Run("startProcessTask, process with error", func(t *testing.T) {
				a := assert.New(t)
				ctx, cancel := context.WithCancel(ctx)
				defer cancel()
				w := sync.WaitGroup{}
				w.Add(1)
				env.m.mode = "hasError"
				go func() {
					n, err := o.processAllOpLog(ctx)
					a.Equal(2, n)
					a.Equal(nil, err)
					w.Done()
				}()
				w.Wait()
				var oplogs []dOpLog
				a.Nil(o.db.OpLog.Find(nil).All(&oplogs))
				a.Len(oplogs, 2)
				if len(oplogs) == 2 {
					a.Len(oplogs[0].Feature, proto.FeatureSize)
					a.Len(oplogs[0].Md5, 32)
					a.Equal(OptatusEvaled, oplogs[0].Status)

					a.Equal("2.jpg", oplogs[1].Key)
					a.Nil(oplogs[1].Feature)
					a.Len(oplogs[1].Md5, 0)
					a.Equal(OptatusEVALERROR, oplogs[1].Status)
				}
			})
		}
	}
}

func TestOpLogUpload(t *testing.T) {
	defer setUpTestEnvLog(t)()
	env := prepareTestEnv(t)
	s := env.s
	o := env.o
	ctx := context.Background()

	user1 := &authstub.Env{UserInfo: authProto.UserInfo{Uid: 111}}
	user2 := &authstub.Env{UserInfo: authProto.UserInfo{Uid: 222}}
	user3 := &authstub.Env{UserInfo: authProto.UserInfo{Uid: 333}}
	user4 := &authstub.Env{UserInfo: authProto.UserInfo{Uid: 444}}

	// user4 30张图片，其它的user imageCntPerUser张
	imageCntPerUser := (proto.KodoBlockFeatureSize*2 + 40) * 3
	imageCntAllUser := imageCntPerUser*3 + 30
	for _, user := range []*authstub.Env{user1, user2, user3, user4} {
		t.Run(fmt.Sprintf("prepare data, add hub, and add image, user %v", user.Uid), func(t *testing.T) {
			a := assert.New(t)
			hubName := fmt.Sprintf("hub-uid-%v", user.Uid)
			err := s.PostHub(ctx, &proto.PostHubReq{Name: hubName, Bucket: "test", Prefix: ""}, user)
			a.Nil(err)

			for i := 0; i < proto.KodoBlockFeatureSize*2+40; i++ {
				if user.Uid == user4.Uid {
					if i == 10 {
						break
					}
				}
				resp, err := s.PostImage(ctx, &proto.PostImageReq{
					Hub: hubName,
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
				}, user)
				a.Equal(3, resp.SuccessCnt)
				a.Nil(err)
			}
		})
	}

	t.Run("check db", func(t *testing.T) {
		a := assert.New(t)
		n, err := s.db.OpLog.Count()
		a.Nil(err)
		a.Equal(imageCntAllUser, n)
	})

	t.Run("startProcessTask, process success", func(t *testing.T) {
		a := assert.New(t)
		ctx, cancel := context.WithCancel(ctx)
		defer cancel()
		w := sync.WaitGroup{}
		w.Add(1)
		env.m.mode = ""
		go func() {
			n, err := o.processAllOpLog(ctx)
			a.Equal(nil, err)
			a.Equal(imageCntAllUser, n)
			w.Done()
		}()
		w.Wait()

		n, err := s.db.OpLog.Find(bson.M{"status": OptatusEvaled, "op": OpKindAdd}).Count()
		a.Nil(err)
		a.Equal(imageCntAllUser, n)
	})

	t.Run("startProcessTask, upload kodo", func(t *testing.T) {
		a := assert.New(t)

		n, err := o.uploadKodo(ctx)
		a.Nil(err)
		a.Equal(proto.KodoBlockFeatureSize*2*3*3, n)

		n, err = s.db.OpLog.Find(bson.M{"status": OptatusEvaled, "op": OpKindAdd}).Count()
		a.Nil(err)
		a.Equal(imageCntAllUser-proto.KodoBlockFeatureSize*2*3*3, n)
		// user4剩30个，其它3个用户每个剩40个
		a.Equal(30+40*3*3, n)

		n, err = s.db.FileMeta.Find(bson.M{}).Count()
		a.Nil(err)
		a.Equal(imageCntAllUser, n)

		n, err = s.db.FileMeta.Find(bson.M{"status": FileMetaStatusOK}).Count()
		a.Nil(err)
		a.Equal(proto.KodoBlockFeatureSize*2*3*3, n)

		n, err = s.db.FileMeta.Find(bson.M{"status": FileMetaStatusInit}).Count()
		a.Nil(err)
		a.Equal(30+40*3*3, n)

		n, err = o.uploadKodo(ctx)
		a.Nil(err)
		a.Equal(0, n)
	})
}
