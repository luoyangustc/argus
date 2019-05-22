package env

import (
	"encoding/base64"
	"os"

	"qiniu.com/argus/test/biz/client"
	biz "qiniu.com/argus/test/biz/util"

	// ENV "qiniu.com/argus/test/biz/env/base"
	"qiniu.com/argus/test/configs"
	"qiniu.com/argus/test/lib/auth"
)

var Env = func() StaticEnv {
	env := os.Getenv("TEST_ENV")
	if env == "product" {
		return StaticEnv{
			GetTSV_: biz.GetTsv,
			GetURIPrivate_: func(name string) string {
				return auth.GetPrivateUrl(
					"http://"+configs.Configs.Atservingprivatebucketz0.Domain+"/"+name,
					configs.Configs.Atservingprivatebucketz0.User.AK, configs.Configs.Atservingprivatebucketz0.User.SK)

				//auth.GetPrivateUrl("http://"+c.Configs.Domain.AtservingprivateDomain+"/"+file, c.Configs.User.AK, c.Configs.User.SK)
			},
			GetURIPrivateV2_: func(name string) string {
				return auth.GetPrivateUrl(
					"http://"+configs.Configs.Atservingprivatebucketz0.Domain+"/"+name,
					configs.Configs.Atservingprivatebucketz0.User.AK, configs.Configs.Atservingprivatebucketz0.User.SK)
			},
			GetURIVideo_: func(name string) string {
				return auth.GetPrivateUrl(
					"http://"+configs.Configs.Atservingprivatebucketz0.Domain+"/"+name,
					configs.Configs.Atservingprivatebucketz0.User.AK, configs.Configs.Atservingprivatebucketz0.User.SK)

				//auth.GetPrivateUrl("http://"+c.Configs.Domain.AtservingprivateDomain+"/"+file, c.Configs.User.AK, c.Configs.User.SK)
			},
			GetURIZ0_: func(name string) string {
				return auth.GetPrivateUrl(
					"http://"+configs.Configs.Atservingprivatebucketz0.Domain+"/"+name,
					configs.Configs.Atservingprivatebucketz0.User.AK, configs.Configs.Atservingprivatebucketz0.User.SK)
			},
			GetClientServing_: func() client.Client {
				return client.NewQiniuClient(configs.StubConfigs.Host.AT_SERVING_GATE, configs.GeneralUser)
			},
			GetClientArgus_: func() client.Client {
				return client.NewQiniuClient(configs.StubConfigs.Host.AT_ARGUS_GATE, configs.GeneralUser)
			},
			GetClientBucket_: func() client.Client {
				return client.NewQiniuClient(configs.Configs.Host.QBOX, configs.GeneralUser)
			},
			GetClientCcpReview_: func() client.Client {
				return client.NewQiniuStubClient(configs.StubConfigs.Host.AT_CCP_REVIEW_GATE, 1, 0)
			},
			GetClientCensor_: func() client.Client {
				return client.NewQiniuClient(configs.StubConfigs.Host.AT_CENSOR_GATE, configs.GeneralUser)
			},
			GetClientArgusVideo_: func() client.Client {
				return client.NewQiniuClient(configs.StubConfigs.Host.ARGUS_VIDEO, configs.GeneralUser)
			},
			GetClientCcpManager_: func() client.Client {
				return client.NewQiniuStubClient(configs.StubConfigs.Host.AT_CCP_MANAGER_GATE, configs.Configs.Users["csbucketuser"].Uid, 1)
			},
			GetClientArgusBjob_: func() client.Client {
				return client.NewQiniuStubClient(configs.StubConfigs.Host.ARGUS_BJOB, 1, 0)
			},
			GetClientCapAdmin_: func() client.Client {
				return client.NewQiniuStubClient(configs.StubConfigs.Host.AT_CAP_ADMIN_GATE, 1, 0)
			},
		}
	} else if env == "dev" || env == "private" {
		return StaticEnv{
			GetTSV_: biz.GetTsv,
			GetURIPrivate_: func(name string) string {
				return auth.GetPrivateUrl(
					"http://"+configs.Configs.Atservingprivatebucketz0.Domain+"/"+name,
					configs.Configs.Atservingprivatebucketz0.User.AK, configs.Configs.Atservingprivatebucketz0.User.SK)
			},
			GetURIPrivateV2_: func(name string) string {
				return auth.GetPrivateUrl(
					"http://"+configs.Configs.Atservingprivatebucketz0.Domain+"/"+name,
					configs.Configs.Atservingprivatebucketz0.User.AK, configs.Configs.Atservingprivatebucketz0.User.SK)
			},
			GetURIVideo_: func(name string) string {
				return auth.GetPrivateUrl(
					"http://"+configs.Configs.Atservingprivatebucketz0.Domain+"/"+name,
					configs.Configs.Atservingprivatebucketz0.User.AK, configs.Configs.Atservingprivatebucketz0.User.SK)
			},
			GetURIZ0_: func(name string) string {
				return auth.GetPrivateUrl(
					"http://"+configs.Configs.Atservingprivatebucketz0.Domain+"/"+name,
					configs.Configs.Atservingprivatebucketz0.User.AK, configs.Configs.Atservingprivatebucketz0.User.SK)
			},
			GetClientServing_: func() client.Client {
				return client.NewQiniuStubClient(configs.StubConfigs.Host.AT_SERVING_GATE, 1, 0)
			},
			GetClientArgus_: func() client.Client {
				return client.NewQiniuStubClient(configs.StubConfigs.Host.AT_ARGUS_GATE, 1, 0)
			},
			GetClientBucket_: func() client.Client {
				return client.NewQiniuClient(configs.Configs.Host.QBOX, configs.GeneralUseronline)
			},
			GetClientCcpReview_: func() client.Client {
				return client.NewQiniuStubClient(configs.StubConfigs.Host.AT_CCP_REVIEW_GATE, 1, 0)
			},
			GetClientCensor_: func() client.Client {
				return client.NewQiniuStubClient(configs.StubConfigs.Host.AT_CENSOR_GATE, 1, 0)
			},
			GetClientArgusVideo_: func() client.Client {
				return client.NewQiniuStubClient(configs.StubConfigs.Host.ARGUS_VIDEO, 1, 0)
			},
			GetClientCcpManager_: func() client.Client {
				return client.NewQiniuStubClient(configs.StubConfigs.Host.AT_CCP_MANAGER_GATE, configs.Configs.Users["csbucketuser"].Uid, 16389)
			},
			GetClientArgusBjob_: func() client.Client {
				return client.NewQiniuStubClient(configs.StubConfigs.Host.ARGUS_BJOB, 1, 0)
			},
			GetClientCapAdmin_: func() client.Client {
				return client.NewQiniuStubClient(configs.StubConfigs.Host.AT_CAP_ADMIN_GATE, 1, 0)
			},
			GetClientArgusDbstorage_: func() client.Client {
				return client.NewQiniuStubClient(configs.StubConfigs.Host.ARGUS_DBSTORAGE, 1, 0)
			},
			GetClientFaceGpu_: func() client.Client {
				return client.NewQiniuStubClient(configs.StubConfigs.Host.FEATURE_GROUP_GPU, 1, 0)
			},
			GetClientFaceCpu_: func() client.Client {
				return client.NewQiniuStubClient(configs.StubConfigs.Host.FEATURE_GROUP_CPU, 1, 0)
			},
		}
	} else {
		return StaticEnv{
			GetTSV_: biz.GetLocal,
			GetURIPrivate_: func(name string) string {
				buf, err := biz.GetLocal(name)
				if err != nil {
					panic(err)
				}
				uriBase64 := "data:application/octet-stream;base64," + base64.StdEncoding.EncodeToString(buf)
				return uriBase64
			},
			GetURIPrivateV2_: func(name string) string {
				return name
			},
			GetURIVideo_: func(name string) string {
				return configs.StubConfigs.Host.SOURCE + name
			},
			GetURIZ0_: func(name string) string {
				return auth.GetPrivateUrl(
					"http://"+configs.Configs.Atservingprivatebucketz0.Domain+"/"+name,
					configs.Configs.Atservingprivatebucketz0.User.AK, configs.Configs.Atservingprivatebucketz0.User.SK)
			},
			GetClientServing_: func() client.Client {
				return client.NewQiniuStubClient(configs.StubConfigs.Host.AT_SERVING_GATE, 1, 0)
			},
			GetClientArgus_: func() client.Client {
				return client.NewQiniuStubClient(configs.StubConfigs.Host.AT_ARGUS_GATE, 1, 0)
			},
			GetClientBucket_: func() client.Client {
				return client.NewQiniuClient(configs.Configs.Host.QBOX, configs.GeneralUser)
			},
			GetClientFaceGpu_: func() client.Client {
				return client.NewQiniuStubClient(configs.StubConfigs.Host.FEATURE_GROUP_GPU, 1, 0)
			},
			GetClientFaceCpu_: func() client.Client {
				return client.NewQiniuStubClient(configs.StubConfigs.Host.FEATURE_GROUP_CPU, 1, 0)
			},
			GetClientCcpReview_: func() client.Client {
				return client.NewQiniuStubClient(configs.StubConfigs.Host.AT_CCP_REVIEW_GATE, 1, 0)
			},
			GetClientCensor_: func() client.Client {
				return client.NewQiniuStubClient(configs.StubConfigs.Host.AT_CENSOR_GATE, 1, 0)
			},
			GetClientArgusVideo_: func() client.Client {
				return client.NewQiniuStubClient(configs.StubConfigs.Host.ARGUS_VIDEO, 1, 0)
			},
		}
	}
}()
