package gate

import (
	"context"
	"github.com/stretchr/testify.v2/assert"
	"testing"

	"github.com/qbox/ke-base/sdk/proto"
)

type mockKebClient struct {
}

func (m mockKebClient) ListResourceSpecs(ctx context.Context) (ret []proto.ResourceSpec, err error) {
	return
}
func (m mockKebClient) PostApp(ctx context.Context, spaceName string, app proto.App) (ret proto.App, err error) {
	return
}
func (m mockKebClient) ListApp(ctx context.Context, spaceName string) (ret []proto.App, err error) {
	return
}
func (m mockKebClient) GetApp(ctx context.Context, spaceName, appName string) (ret proto.App, err error) {
	return
}
func (m mockKebClient) DeleteApp(ctx context.Context, spaceName, appName string) (err error) {
	return
}
func (m mockKebClient) PostService(ctx context.Context, spaceName, appName string, svc proto.MicroService) (ret proto.MicroService, err error) {
	return
}
func (m mockKebClient) ListService(ctx context.Context, spaceNamee, appName string) (ret []proto.MicroService, err error) {
	return
}
func (m mockKebClient) GetService(ctx context.Context, spaceName, appName, svcName string) (ret proto.MicroService, err error) {
	return
}
func (m mockKebClient) DeleteService(ctx context.Context, spaceName, appName, svcName string) (err error) {
	return
}
func (m mockKebClient) UpgradeService(ctx context.Context, spaceName, appName, svcName string, svc proto.MicroServiceUpgradeArgs) (ret proto.MicroService, err error) {
	return
}
func (m mockKebClient) UpdateServicePorts(ctx context.Context, spaceName, appName, svcName string, svc proto.MicroServiceUpdatePortsArgs) (ret proto.MicroService, err error) {
	return
}
func (m mockKebClient) ScaleService(ctx context.Context, spaceName, appName, svcName string, num int32) (err error) {
	return
}
func (m mockKebClient) StartService(ctx context.Context, spaceName, appName, svcName string) (err error) {
	return
}
func (m mockKebClient) StopService(ctx context.Context, spaceName, appName, svcName string) (err error) {
	return
}
func (m mockKebClient) ListServiceRevisions(ctx context.Context, spaceName, appName, svcName string) (ret []proto.MicroserviceRevision, err error) {
	return
}
func (m mockKebClient) RollbackService(ctx context.Context, spaceName, appName, svcName string, revision int32) (err error) {
	return
}

func (m mockKebClient) PostServiceV2(ctx context.Context, spaceName string, svc proto.MicroService) (ret proto.MicroService, err error) {
	return
}
func (m mockKebClient) ListServiceV2(ctx context.Context, spaceName string) (ret []proto.MicroService, err error) {
	ret = append(ret, proto.MicroService{
		Name: "test",
	})
	return
}
func (m mockKebClient) GetServiceV2(ctx context.Context, spaceName, svcName string) (ret proto.MicroService, err error) {
	return
}
func (m mockKebClient) DeleteServiceV2(ctx context.Context, spaceName, svcName string) (err error) {
	return
}
func (m mockKebClient) UpgradeServiceV2(ctx context.Context, spaceName, svcName string, svc proto.MicroServiceUpgradeArgs) (ret proto.MicroService, err error) {
	return
}
func (m mockKebClient) UpdateServicePortsV2(ctx context.Context, spaceName, svcName string, svc proto.MicroServiceUpdatePortsArgs) (ret proto.MicroService, err error) {
	return
}
func (m mockKebClient) ScaleServiceV2(ctx context.Context, spaceName, svcName string, num int32) (err error) {
	return
}
func (m mockKebClient) StartServiceV2(ctx context.Context, spaceName, svcName string) (err error) {
	return
}
func (m mockKebClient) StopServiceV2(ctx context.Context, spaceName, svcName string) (err error) {
	return
}
func (m mockKebClient) ListServiceRevisionsV2(ctx context.Context, spaceName, svcName string) (ret []proto.MicroserviceRevision, err error) {
	return
}
func (m mockKebClient) RollbackServiceV2(ctx context.Context, spaceName, svcName string, revision int32) (err error) {
	return
}

func TestRouter(t *testing.T) {
	cf := RouterConfig{
		AppsConf: struct {
			Workspace string `json:"workspace"`
			Port      string `json:"port"`
			AppPrefix string `json:"app_prefix"`
		}{},
	}
	route := NewRouter(cf, &mockKebClient{})
	path := route.Match(context.Background(), "test")
	assert.Equal(t, "", path)
}
