package service

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"testing"
	"time"

	"gopkg.in/mgo.v2"

	"qiniu.com/argus/feature_group_private/proto"
	"qiniu.com/argus/feature_group_private/search"

	mgoutil "github.com/qiniu/db/mgoutil.v3"
	"github.com/stretchr/testify/assert"
)

func TestImageGroups(t *testing.T) {
	var featureHost = runMockServer()
	if os.Getenv("NOMOCK") != "" {
		featureHost = "http://ava-serving-gate.cs.cg.dora-internal.qiniu.io:5001"
	}
	a := assert.New(t)
	ctx := context.Background()
	var (
		MGO_HOST = "mongodb://127.0.0.1"
		MGO_DB   = "feature_group_private_image_test"
	)

	sess, err := mgo.Dial(MGO_HOST)
	a.Nil(err)
	a.Nil(sess.DB(MGO_DB).DropDatabase())
	baseConfig := BaseGroupsConfig{
		MgoConfig: mgoutil.Config{
			Host: MGO_HOST,
			DB:   MGO_DB,
			Mode: "strong",
		},
		Sets: search.Config{
			Dimension: 2048,
			Precision: 4,
			Version:   0,
			DeviceID:  0,
			BlockSize: 2048 * 4 * 10,
			BlockNum:  100,
			BatchSize: 5,
		},
		CollSessionPoolLimit: 50,
	}

	bs, err := NewBaseGroups(ctx, baseConfig, "")
	a.Nil(err)

	s, err := NewImageGroups(ctx, bs, ImageGroupsConfig{
		BaseGroupsConfig:    baseConfig,
		ImageFeatureHost:    featureHost,
		ImageFeatureTimeout: time.Duration(15) * time.Second,
	})
	a.Nil(err)

	t.Run("插入一张图片可以搜索出来", func(t *testing.T) {
		groupName := proto.GroupName("group1")
		err = s.New(ctx, groupName, proto.GroupConfig{
			Capacity: 100,
		})
		a.Nil(err)
		g, err := s.Get(ctx, groupName)
		a.Nil(err)
		gs, err := s.All(ctx)
		a.Nil(err)
		a.Len(gs, 1)
		a.Equal(gs[0], groupName)
		img1 := proto.ImageURI("http://q.hi-hi.cn/1.png")
		a.Nil(g.AddImage(ctx, proto.Image{
			ID:   proto.FeatureID("id1"),
			URI:  img1,
			Tag:  "tag1",
			Desc: json.RawMessage("desc1"),
		}))
		ss, err := g.SearchImage(ctx, 0, 100, img1)
		a.Nil(err)
		a.Len(ss, 1)
		a.Len(ss[0], 1)
		a.Equal(ss[0][0].ID, proto.FeatureID("id1"))
	})

	t.Run("CRUD多张图片", func(t *testing.T) {
		groupName := proto.GroupName("group2")
		err = s.New(ctx, groupName, proto.GroupConfig{
			Capacity: 100,
		})
		a.Nil(err)
		g, err := s.Get(ctx, groupName)
		a.Nil(err)
		img1 := proto.ImageURI("http://q.hi-hi.cn/1.png")
		img2 := proto.ImageURI("https://www.qiniu.com/assets/icon-controllale@2x-47c22ae3192d5b1a26f8ccb4852d67ea8a1d10d5ab357bda51959edabdab1237.png")
		img3 := proto.ImageURI("https://odum9helk.qnssl.com/FjkrNUuQ8bTqaPAEsgGZBAYMi5qS")
		a.Nil(g.AddImage(ctx, proto.Image{
			ID:   proto.FeatureID("id1"),
			URI:  img1,
			Tag:  "tag1",
			Desc: json.RawMessage("desc1"),
		}, proto.Image{
			ID:   proto.FeatureID("id2"),
			URI:  img2,
			Tag:  "tag2",
			Desc: json.RawMessage("desc2"),
		}, proto.Image{
			ID:   proto.FeatureID("id3"),
			URI:  img3,
			Tag:  "tag3",
			Desc: json.RawMessage("desc3"),
		}, proto.Image{
			ID:   proto.FeatureID("id4"),
			URI:  img3,
			Tag:  "tag3",
			Desc: json.RawMessage("desc4"),
		}))

		ss, err := g.SearchImage(ctx, 0.99, 100, img1)
		fmt.Println(ss)
		a.Nil(err)
		a.Len(ss, 1)
		a.Len(ss[0], 1)
		a.Equal(ss[0][0].ID, proto.FeatureID("id1"))

		ss, err = g.SearchImage(ctx, 0.99, 100, img2)
		a.Nil(err)
		a.Len(ss, 1)
		a.Len(ss[0], 1)
		a.Equal(ss[0][0].ID, proto.FeatureID("id2"))

		ss, err = g.SearchImage(ctx, 0.99, 100, img3)
		a.Nil(err)
		a.Len(ss, 1)
		a.Len(ss[0], 2)
		a.Contains([]proto.FeatureID{"id3", "id4"}, ss[0][0].ID)
		a.Contains([]proto.FeatureID{"id3", "id4"}, ss[0][1].ID)
	})

	t.Run("根据Tag过滤图片", func(t *testing.T) {
		groupName := proto.GroupName("group3")
		err = s.New(ctx, groupName, proto.GroupConfig{
			Capacity: 100,
		})
		a.Nil(err)
		g, err := s.Get(ctx, groupName)
		a.Nil(err)
		img1 := proto.ImageURI("http://q.hi-hi.cn/1.png")
		img2 := proto.ImageURI("https://www.qiniu.com/assets/icon-controllale@2x-47c22ae3192d5b1a26f8ccb4852d67ea8a1d10d5ab357bda51959edabdab1237.png")
		img3 := proto.ImageURI("https://odum9helk.qnssl.com/FjkrNUuQ8bTqaPAEsgGZBAYMi5qS")
		a.Nil(g.AddImage(ctx, proto.Image{
			ID:   proto.FeatureID("id1"),
			URI:  img1,
			Tag:  "tag",
			Desc: json.RawMessage("desc1"),
		}, proto.Image{
			ID:   proto.FeatureID("id2"),
			URI:  img2,
			Tag:  "tag",
			Desc: json.RawMessage("desc2"),
		}, proto.Image{
			ID:   proto.FeatureID("id3"),
			URI:  img3,
			Tag:  "tag",
			Desc: json.RawMessage("desc3"),
		}, proto.Image{
			ID:   proto.FeatureID("id4"),
			URI:  img3,
			Tag:  "tag",
			Desc: json.RawMessage("desc4"),
		}))

		images, marker, err := g.ListImage(ctx, "", "", 0)
		a.Nil(err)
		a.Len(images, 4)
		a.Empty(marker)

		images, marker, err = g.ListImage(ctx, "not-exist-tag", "", 0)
		a.Nil(err)
		a.Len(images, 0)
		a.Empty(marker)

		images, marker, err = g.ListImage(ctx, "tag", "", 0)
		a.Nil(err)
		a.Empty(marker)
		a.Len(images, 4)
		a.Equal([]proto.FeatureID{"id1", "id2", "id3", "id4"}, []proto.FeatureID{images[0].ID, images[1].ID, images[2].ID, images[3].ID})

		images, marker, err = g.ListImage(ctx, "tag", "", 3)
		a.Nil(err)
		a.NotEmpty(marker)
		a.Len(images, 3)
		a.Equal([]proto.FeatureID{"id1", "id2", "id3"}, []proto.FeatureID{images[0].ID, images[1].ID, images[2].ID})

		images, marker, err = g.ListImage(ctx, "tag", marker, 3)
		a.Nil(err)
		a.Empty(marker)
		if a.Len(images, 1) {
			a.Equal(proto.FeatureID("id4"), images[0].ID)
		}
	})
}
