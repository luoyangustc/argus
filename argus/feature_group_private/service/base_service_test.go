package service

import (
	"context"
	"net/http/httptest"
	"testing"

	restrpc "github.com/qiniu/http/restrpc.v1"
	"github.com/stretchr/testify/assert"
	feature_group "qiniu.com/argus/feature_group_private"
	"qiniu.com/argus/feature_group_private/manager"
	"qiniu.com/argus/feature_group_private/proto"
)

type mockBaseGroups struct {
	Groups []proto.GroupName
}

var _ feature_group.IGroups = &mockBaseGroups{}

func (s *mockBaseGroups) New(ctx context.Context, internal bool, group proto.GroupName, cfg proto.GroupConfig) (err error) {
	return nil
}

func (s *mockBaseGroups) Get(ctx context.Context, group proto.GroupName) (feature_group.IGroup, error) {
	for _, g := range s.Groups {
		if g == group {
			return &_BaseGroup{}, nil
		}
	}
	return nil, manager.ErrGroupNotExist
}
func (s *mockBaseGroups) All(ctx context.Context) ([]proto.GroupName, error) {
	return s.Groups, nil
}

func TestBaseService(t *testing.T) {
	var (
		bg     = &mockBaseGroups{}
		ctx    = context.Background()
		config BaseGroupsConfig
		a      = assert.New(t)
		env    = &restrpc.Env{
			W:   httptest.NewRecorder(),
			Req: httptest.NewRequest("GET", "/v1/face/group", nil),
		}
	)

	service, err := NewBaseService(ctx, bg, config)
	a.Nil(err)
	a.NotNil(service)

	args := &struct {
		CmdArgs         []string
		Config          proto.GroupConfig `json:"config"`
		ClusterInternal bool              `json:"cluster_internal"`
	}{
		CmdArgs: []string{""},
		Config: proto.GroupConfig{
			Capacity: 100,
		},
		ClusterInternal: false,
	}

	t.Run("创建group name为空", func(t *testing.T) {
		err := service.PostGroups_(ctx, args, env)
		a.NotNil(err)
		a.Equal("invalid arguments", err.Error())
	})
	t.Run("创建group时name不合法", func(t *testing.T) {
		args.CmdArgs = []string{"!@#!@#!@#adfa5"}
		err := service.PostGroups_(ctx, args, env)
		a.NotNil(err)
		a.Equal("invalid group name", err.Error())

		args.CmdArgs = []string{"aa"}
		err = service.PostGroups_(ctx, args, env)
		a.NotNil(t, err)
		a.Equal("invalid group name", err.Error())

		args.CmdArgs = []string{"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"}
		a.Equal(33, len(args.CmdArgs[0]))
		err = service.PostGroups_(ctx, args, env)
		a.NotNil(err)
		a.Equal("invalid group name", err.Error())
	})
	t.Run("创建Group", func(t *testing.T) {
		args.CmdArgs = []string{"group1"}
		a.Nil(service.PostGroups_(ctx, args, env))
	})
	t.Run("重复创建Group", func(t *testing.T) {
		args.CmdArgs = []string{"group1"}
		bg.Groups = append(bg.Groups, proto.GroupName("group1"))
		err := service.PostGroups_(ctx, args, env)
		a.NotNil(err)
		a.Equal("group already exist", err.Error())
	})
	t.Run("Capacity小于零", func(t *testing.T) {
		args.CmdArgs = []string{"group2"}
		args.Config.Capacity = -1
		err := service.PostGroups_(ctx, args, env)
		a.NotNil(err)
		a.Equal("invalid group capacity", err.Error())
	})

	arg := &struct{ CmdArgs []string }{CmdArgs: []string{""}}
	t.Run("查询全部Group name", func(t *testing.T) {
		ret, err := service.GetGroups(ctx, arg, env)
		a.Nil(err)
		a.Equal(1, len(ret.Groups))
		a.Equal(proto.GroupName("group1"), ret.Groups[0])
	})
}
