package client

import (
	"context"
	etcd "github.com/coreos/etcd/clientv3"
	"github.com/stretchr/testify.v2/assert"
	"testing"
	"time"
)

const PREFIX = "/temp/ava/atserving/config"

var HOST = []string{"localhost:2379"}

func cleanDb(t *testing.T) {
	assert := assert.New(t)
	cli, err := etcd.New(etcd.Config{Endpoints: HOST, DialTimeout: time.Second})
	assert.Nil(err)
	_, err = cli.Delete(context.Background(), PREFIX, etcd.WithPrefix())
	assert.Nil(err)
}

func TestPostRelease(t *testing.T) {
	cleanDb(t)
	assert := assert.New(t)
	client, err := New(Config{
		KeyPrefix: PREFIX,
		Etcd: etcd.Config{
			Endpoints:   HOST,
			DialTimeout: time.Second,
		},
	})
	assert.Nil(err)
	ctx := context.Background()
	{
		resp1, err := client.PostRelease(ctx, PostReleaseArgs{
			App: "ccccc",
			MetaData: MetaData{
				TarURI: "xxx",
			},
			Desc: "step1",
		})
		assert.Nil(err)
		resp2, err := client.GetRelease(ctx, "ccccc", resp1.Version)
		assert.Equal(resp2.Desc, "step1")
	}
	{
		resp1, err := client.PostRelease(ctx, PostReleaseArgs{
			App: "ccccc",
			MetaData: MetaData{
				TarURI: "xxx111",
			},
			Desc: "step2",
		})
		assert.Nil(err)
		resp2, err := client.GetRelease(ctx, "ccccc", resp1.Version)
		assert.Equal(resp2.Desc, "step2")
	}
	var version string
	{
		resp, err := client.PostRelease(ctx, PostReleaseArgs{
			App: "aaaa",
			MetaData: MetaData{
				TarURI: "xxx111",
			},
			Desc: "step3",
		})
		assert.Nil(err)
		version = resp.Version
	}
	{
		resp, err := client.GetRelease(ctx, "aaaa", version)
		assert.Nil(err)
		assert.Equal(resp.MetaData.TarURI, "xxx111")
		assert.True(time.Since(resp.CreateTime) < time.Second)
	}
	{
		resp, err := client.ListRelease(ctx, "aaaa")
		assert.Nil(err)
		assert.Len(resp, 1)
		assert.Equal((resp)[0].Desc, "step3")
	}
}

func Test_genVersion(t *testing.T) {
	assert := assert.New(t)
	assert.Len(genVersion(), 19)
}
