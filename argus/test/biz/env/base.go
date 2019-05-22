package env

import (
	"qiniu.com/argus/test/biz/client"
)

type IEnv interface {
	GetTSV(string) ([]byte, error)

	GetURIPublic(string) string
	GetURIPrivate(string) string
	GetURIPrivateV2(string) string
	GetURIZ0(string) string
	GetURIZ1(string) string

	GetClientServing() client.Client
	GetClientArgus() client.Client
	GetClientFaceGpu() client.Client
	GetClientFaceCpu() client.Client
	GetClientBucket() client.Client
	GetClientCensor() client.Client
	GetClientCcpManager() client.Client
}

var _ IEnv = StaticEnv{}

type StaticEnv struct {
	GetTSV_ func(string) ([]byte, error)

	GetURIPublic_            func(string) string
	GetURIPrivate_           func(string) string
	GetURIPrivateV2_         func(string) string
	GetURIZ0_                func(string) string
	GetURIZ1_                func(string) string
	GetURILivePrivate_       func(string) string
	GetURIVideo_             func(string) string
	GetHOOKURIPrivate_       func(string) string
	GetClientServing_        func() client.Client
	GetClientArgus_          func() client.Client
	GetClientFaceGpu_        func() client.Client
	GetClientFaceCpu_        func() client.Client
	GetClientBucket_         func() client.Client
	GetClientLive_           func() client.Client
	GetClientLiveFace_       func() client.Client
	GetClientCensor_         func() client.Client
	GetClientCcpManager_     func() client.Client
	GetClientArgusBjob_      func() client.Client
	GetClientArgusVideo_     func() client.Client
	GetClientArgusDbstorage_ func() client.Client
	GetClientCcpReview_      func() client.Client
	GetClientCapAdmin_       func() client.Client
}

func (env StaticEnv) GetTSV(name string) ([]byte, error) { return env.GetTSV_(name) }
func (env StaticEnv) GetURIPublic(name string) string    { return env.GetURIPublic_(name) }
func (env StaticEnv) GetURIPrivate(name string) string   { return env.GetURIPrivate_(name) }
func (env StaticEnv) GetURIPrivateV2(name string) string { return env.GetURIPrivateV2_(name) }
func (env StaticEnv) GetURIZ0(name string) string        { return env.GetURIZ0_(name) }
func (env StaticEnv) GetURIZ1(name string) string        { return env.GetURIZ1_(name) }
func (env StaticEnv) GetClientServing() client.Client    { return env.GetClientServing_() }
func (env StaticEnv) GetClientArgus() client.Client      { return env.GetClientArgus_() }
func (env StaticEnv) GetClientFaceGpu() client.Client    { return env.GetClientFaceGpu_() }
func (env StaticEnv) GetClientFaceCpu() client.Client    { return env.GetClientFaceCpu_() }
func (env StaticEnv) GetClientBucket() client.Client     { return env.GetClientBucket_() }
func (env StaticEnv) GetClientCcpReview() client.Client  { return env.GetClientCcpReview_() }
func (env StaticEnv) GetURIVideo(name string) string     { return env.GetURIVideo_(name) }
func (env StaticEnv) GetURILivePrivate(livename string) string {
	return env.GetURILivePrivate_(livename)
}
func (env StaticEnv) GetHOOKURLPrivate(port string) string   { return env.GetHOOKURIPrivate_(port) }
func (env StaticEnv) GetClientLive() client.Client           { return env.GetClientLive_() }
func (env StaticEnv) GetClientLiveFace() client.Client       { return env.GetClientLiveFace_() }
func (env StaticEnv) GetClientCensor() client.Client         { return env.GetClientCensor_() }
func (env StaticEnv) GetClientCcpManager() client.Client     { return env.GetClientCcpManager_() }
func (env StaticEnv) GetClientArgusBjob() client.Client      { return env.GetClientArgusBjob_() }
func (env StaticEnv) GetClientArgusVideo() client.Client     { return env.GetClientArgusVideo_() }
func (env StaticEnv) GetClientCapAdmin() client.Client       { return env.GetClientCapAdmin_() }
func (env StaticEnv) GetClientArgusDbstorage() client.Client { return env.GetClientArgusDbstorage_() }
