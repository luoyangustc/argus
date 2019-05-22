package service

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"testing"
	"time"

	mgo "gopkg.in/mgo.v2"
	"qiniu.com/argus/feature_group_private/proto"
	"qiniu.com/argus/feature_group_private/search"

	mgoutil "github.com/qiniu/db/mgoutil.v3"
	"github.com/stretchr/testify/assert"
)

func TestFaceGroups(t *testing.T) {
	t.Skip()
	var featureHost = runMockServer()
	if os.Getenv("NOMOCK") != "" {
		featureHost = "http://ava-serving-gate.cs.cg.dora-internal.qiniu.io:5001"
	}
	a := assert.New(t)
	ctx := context.Background()
	var (
		MGO_HOST = "mongodb://127.0.0.1"
		MGO_DB   = "face_group_private"
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
			Dimension: 512,
			Precision: 4,
			Version:   0,
			DeviceID:  0,
			BlockSize: 512 * 4 * 100,
			BlockNum:  100,
			BatchSize: 5,
		},
		CollSessionPoolLimit: 50,
	}
	bs, err := NewBaseGroups(ctx, baseConfig, "")
	a.Nil(err)

	s, err := NewFaceGroups(ctx, bs, FaceGroupsConfig{
		BaseGroupsConfig:   baseConfig,
		FaceFeatureHost:    featureHost,
		FaceFeatureTimeout: time.Duration(15) * time.Second,
	})
	a.Nil(err)

	groupName := proto.GroupName("")
	_, err = s.Get(ctx, groupName)
	assert.Equal(t, "Invalid Group Name", err.Error())

	_, err = s.baseGroups.Get(ctx, "this_is_no_exit_groupname")
	assert.Equal(t, "group is not exist", err.Error())

	t.Run("插入一张指定坐标的人脸或非人脸可以搜索出来", func(t *testing.T) {
		groupName := proto.GroupName("group0")
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
		img1 := proto.ImageURI("http://img1.gtimg.com/fashion/pics/hv1/211/150/1574/102387811.jpg")

		_, _, err = g.AddFace(ctx, false, proto.Image{
			ID:   proto.FeatureID("id1"),
			URI:  img1,
			Tag:  "wawa1",
			Desc: json.RawMessage("desc1"),
			BoundingBox: proto.BoundingBox{
				Pts: [][2]int{[2]int{108, 55}, [2]int{391, 55}, [2]int{391, 357}, [2]int{108, 357}},
			},
		})
		a.Nil(err)

		// 重复插入图片报错
		_, _, err = g.AddFace(ctx, false, proto.Image{
			ID:   proto.FeatureID("id1"),
			URI:  img1,
			Tag:  "tag1",
			Desc: json.RawMessage("desc1"),
			BoundingBox: proto.BoundingBox{
				Pts: [][2]int{[2]int{108, 55}, [2]int{391, 55}, [2]int{391, 357}, [2]int{108, 357}},
			},
		})
		a.NotNil(err)

		data := make([]proto.Data, 1)
		data[0].URI = img1
		fvss, faceBoxess, err := s.DetectAndFetchFeature(ctx, false, data)
		a.Nil(err)
		a.Len(fvss, 1)
		a.Len(faceBoxess, 1)
		a.Len(fvss[0], 1)
		a.Len(faceBoxess[0], 1)

		ss, err := g.SearchFace(ctx, 0.99, 100, fvss)
		a.Nil(err)
		a.Len(ss, 1)
		a.Len(ss[0], 1)
		a.Len(ss[0][0].Faces, 1)
		a.Equal(ss[0][0].Faces[0].ID, proto.FeatureID("id1"))
	})

	t.Run("插入一张人脸可以搜索出来", func(t *testing.T) {
		groupName := proto.GroupName("group1")
		err = s.New(ctx, groupName, proto.GroupConfig{
			Capacity: 100,
		})
		a.Nil(err)
		g, err := s.Get(ctx, groupName)
		a.Nil(err)
		gs, err := s.All(ctx)
		a.Nil(err)
		a.Len(gs, 2)
		a.Equal(gs[1], groupName)
		img1 := proto.ImageURI("http://img1.gtimg.com/fashion/pics/hv1/211/150/1574/102387811.jpg")

		_, _, err = g.AddFace(ctx, false, proto.Image{
			ID:   proto.FeatureID("id1"),
			URI:  img1,
			Tag:  "tag1",
			Desc: json.RawMessage("desc1"),
		})
		a.Nil(err)

		// 重复插入图片报错
		_, _, err = g.AddFace(ctx, false, proto.Image{
			ID:   proto.FeatureID("id1"),
			URI:  img1,
			Tag:  "tag1",
			Desc: json.RawMessage("desc1"),
		})
		a.NotNil(err)

		data := make([]proto.Data, 1)
		data[0].URI = img1
		fvss, faceBoxes, err := s.DetectAndFetchFeature(ctx, true, data)
		a.Nil(err)
		a.Len(fvss, 1)
		a.Len(faceBoxes, 1)
		a.Len(fvss[0], 1)
		a.Len(faceBoxes[0], 1)
		fmt.Println(faceBoxes)

		ss, err := g.SearchFace(ctx, 0.99, 100, fvss)
		a.Nil(err)
		a.Len(ss, 1)
		a.Len(ss[0], 1)
		a.Len(ss[0][0].Faces, 1)
		a.Equal(ss[0][0].Faces[0].ID, proto.FeatureID("id1"))

	})

	t.Run("插入一张没有人脸的图片报错", func(t *testing.T) {
		groupName := proto.GroupName("group2")
		err = s.New(ctx, groupName, proto.GroupConfig{
			Capacity: 100,
		})
		a.Nil(err)
		g, err := s.Get(ctx, groupName)
		a.Nil(err)
		img1 := proto.ImageURI("https://odum9helk.qnssl.com/FjkrNUuQ8bTqaPAEsgGZBAYMi5qS")
		_, _, err = g.AddFace(ctx, false, proto.Image{
			ID:   proto.FeatureID("image-with-no-face"),
			URI:  img1,
			Tag:  "tag1",
			Desc: json.RawMessage("desc1"),
		})
		a.NotNil(err)

		data := make([]proto.Data, 1)
		data[0].URI = img1
		fvss, faceBoxess, err := s.DetectAndFetchFeature(ctx, false, data)
		a.Len(fvss, 1)
		a.Len(faceBoxess, 1)
		a.Len(fvss[0], 0)
		a.Len(faceBoxess[0], 0)
		a.Nil(err)

		ss, err := g.SearchFace(ctx, 0, 100, fvss)
		a.Nil(err)
		a.Len(ss, 1)
		a.Len(ss[0], 0)

		g2, err := bs.Get(ctx, groupName)
		a.Nil(err)
		n, err := g2.Count(ctx)
		a.Nil(err)
		a.Equal(n, 0)

	})

	t.Run("CRUD多张图片", func(t *testing.T) {
		groupName := proto.GroupName("group3")
		err = s.New(ctx, groupName, proto.GroupConfig{
			Capacity: 100,
		})
		a.Nil(err)
		g, err := s.Get(ctx, groupName)
		a.Nil(err)
		img1 := proto.ImageURI("http://q.hi-hi.cn/1.png")
		img2 := proto.ImageURI("http://img.leha.com/0125/bfea69425e548.jpg")
		img3 := proto.ImageURI("http://img1.gtimg.com/fashion/pics/hv1/211/150/1574/102387811.jpg")
		boxes, _, err := g.AddFace(ctx, false, proto.Image{
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
		},
		)
		a.Equal(4, len(boxes))
		a.Nil(err)

		data := make([]proto.Data, 1)
		data[0].URI = img1
		fvss, faceBoxess, err := s.DetectAndFetchFeature(ctx, false, data)
		a.Nil(err)
		a.Len(fvss, 1)
		a.Len(faceBoxess, 1)
		a.Len(fvss[0], 1)
		a.Len(faceBoxess[0], 1)

		ss, err := g.SearchFace(ctx, 0.99, 100, fvss)
		a.Nil(err)
		a.Len(ss, 1)
		a.Len(ss[0], 1)
		a.Len(ss[0][0].Faces, 1)
		a.Equal(ss[0][0].Faces[0].ID, proto.FeatureID("id1"))

		data = make([]proto.Data, 1)
		data[0].URI = img2
		fvss, faceBoxess, err = s.DetectAndFetchFeature(ctx, false, data)
		a.Nil(err)
		a.Len(fvss, 1)
		a.Len(faceBoxess, 1)
		a.Len(fvss[0], 1)
		a.Len(faceBoxess[0], 1)

		ss, err = g.SearchFace(ctx, 0.99, 100, fvss)
		a.Nil(err)
		a.Len(ss, 1)
		a.Len(ss[0], 1)
		a.Len(ss[0][0].Faces, 1)
		a.Equal(ss[0][0].Faces[0].ID, proto.FeatureID("id2"))

		data = make([]proto.Data, 1)
		data[0].URI = img3
		fvss, faceBoxess, err = s.DetectAndFetchFeature(ctx, false, data)
		a.Nil(err)
		a.Len(fvss, 1)
		a.Len(faceBoxess, 1)
		a.Len(fvss[0], 1)
		a.Len(faceBoxess[0], 1)

		ss, err = g.SearchFace(ctx, 0.99, 100, fvss)
		a.Nil(err)
		a.Len(ss, 1)
		a.Len(ss[0], 1)
		a.Len(ss[0][0].Faces, 2)
		a.Contains([]proto.FeatureID{"id3", "id4"}, ss[0][0].Faces[0].ID)
		a.Contains([]proto.FeatureID{"id3", "id4"}, ss[0][0].Faces[1].ID)

	})

	t.Run("根据Tag过滤图片", func(t *testing.T) {
		groupName := proto.GroupName("group4")
		err = s.New(ctx, groupName, proto.GroupConfig{
			Capacity: 100,
		})
		a.Nil(err)
		g, err := s.Get(ctx, groupName)
		a.Nil(err)
		img1 := proto.ImageURI("http://q.hi-hi.cn/1.png")
		img2 := proto.ImageURI("http://img.leha.com/0125/bfea69425e548.jpg")
		img3 := proto.ImageURI("http://img1.gtimg.com/fashion/pics/hv1/211/150/1574/102387811.jpg")
		_, _, err = g.AddFace(ctx, false, proto.Image{
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
		})
		a.Nil(err)

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

	t.Run("更新一张人脸并搜索出来", func(t *testing.T) {
		groupName := proto.GroupName("group5")
		err = s.New(ctx, groupName, proto.GroupConfig{
			Capacity: 100,
		})
		a.Nil(err)
		g, err := s.Get(ctx, groupName)
		a.Nil(err)
		img1 := proto.ImageURI("http://img1.gtimg.com/fashion/pics/hv1/211/150/1574/102387811.jpg")

		_, _, err = g.AddFace(ctx, false, proto.Image{
			ID:   proto.FeatureID("id1"),
			URI:  img1,
			Tag:  "tag1",
			Desc: json.RawMessage("desc1"),
		})
		a.Nil(err)

		data := make([]proto.Data, 1)
		data[0].URI = img1
		fvss, faceBoxess, err := s.DetectAndFetchFeature(ctx, false, data)
		a.Nil(err)
		a.Len(fvss, 1)
		a.Len(faceBoxess, 1)
		a.Len(fvss[0], 1)
		a.Len(faceBoxess[0], 1)

		ss, err := g.SearchFace(ctx, 0.99, 100, fvss)
		a.Nil(err)
		a.Len(ss, 1)
		a.Len(ss[0], 1)
		a.Len(ss[0][0].Faces, 1)
		a.Equal(ss[0][0].Faces[0].ID, proto.FeatureID("id1"))
		a.Equal(ss[0][0].Faces[0].Tag, proto.FeatureTag("tag1"))

		a.Nil(g.UpdateFace(ctx, false, proto.Image{
			ID:   proto.FeatureID("id1"),
			URI:  img1,
			Tag:  "tag2",
			Desc: json.RawMessage("desc1"),
		}))

		data = make([]proto.Data, 1)
		data[0].URI = img1
		fvss, faceBoxess, err = s.DetectAndFetchFeature(ctx, false, data)
		a.Nil(err)
		a.Len(fvss, 1)
		a.Len(faceBoxess, 1)
		a.Len(fvss[0], 1)
		a.Len(faceBoxess[0], 1)

		ss, err = g.SearchFace(ctx, 0.99, 100, fvss)
		a.Nil(err)
		a.Len(ss, 1)
		a.Len(ss[0], 1)
		a.Len(ss[0][0].Faces, 1)
		fmt.Println(ss[0][0].Faces[0].ID)
		a.Equal(ss[0][0].Faces[0].ID, proto.FeatureID("id1"))
		a.Equal(ss[0][0].Faces[0].Tag, proto.FeatureTag("tag2"))
	})

	t.Run("开启search_cache, 插入一张人脸可以搜索出来", func(t *testing.T) {
		s.config.SearchCache = true
		groupName := proto.GroupName("group6")
		err = s.New(ctx, groupName, proto.GroupConfig{
			Capacity: 100,
		})
		a.Nil(err)
		g, err := s.Get(ctx, groupName)
		a.Nil(err)
		img1 := proto.ImageURI("http://img1.gtimg.com/fashion/pics/hv1/211/150/1574/102387811.jpg")

		_, _, err = g.AddFace(ctx, false, proto.Image{
			ID:   proto.FeatureID("id1"),
			URI:  img1,
			Tag:  "tag1",
			Desc: json.RawMessage("desc1"),
		})
		a.Nil(err)

		data := make([]proto.Data, 1)
		data[0].URI = img1
		fvss, faceBoxess, err := s.DetectAndFetchFeature(ctx, false, data)
		a.Nil(err)
		a.Len(fvss, 1)
		a.Len(faceBoxess, 1)
		a.Len(fvss[0], 1)
		a.Len(faceBoxess[0], 1)

		ss, err := g.SearchFace(ctx, 0, 100, fvss)
		a.Nil(err)
		a.Len(ss, 1)
		a.Len(ss[0], 1)
		a.Len(ss[0][0].Faces, 1)
		a.Equal(ss[0][0].Faces[0].ID, proto.FeatureID("id1"))
	})

	t.Run("开启search_cache, 插入一张非人脸报错", func(t *testing.T) {
		s.config.SearchCache = true
		groupName := proto.GroupName("group7")
		err = s.New(ctx, groupName, proto.GroupConfig{
			Capacity: 100,
		})
		a.Nil(err)
		g, err := s.Get(ctx, groupName)
		a.Nil(err)
		img1 := proto.ImageURI("https://odum9helk.qnssl.com/FjkrNUuQ8bTqaPAEsgGZBAYMi5qS")

		_, _, err = g.AddFace(ctx, false, proto.Image{
			ID:   proto.FeatureID("id1"),
			URI:  img1,
			Tag:  "tag1",
			Desc: json.RawMessage("desc1"),
		})
		a.NotNil(err)

		data := make([]proto.Data, 1)
		data[0].URI = img1
		fvss, faceBoxess, err := s.DetectAndFetchFeature(ctx, false, data)
		a.Nil(err)
		a.Len(fvss, 1)
		a.Len(faceBoxess, 1)
		a.Len(fvss[0], 0)
		a.Len(faceBoxess[0], 0)

		ss, err := g.SearchFace(ctx, 0, 100, fvss)
		a.Nil(err)
		a.Len(ss, 1)
		a.Len(ss[0], 0)
	})

	t.Run("插入一张人脸或非人脸指定坐标搜索出来", func(t *testing.T) {
		groupName := proto.GroupName("group8")
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
		img1 := proto.ImageURI("http://img1.gtimg.com/fashion/pics/hv1/211/150/1574/102387811.jpg")

		_, _, err = g.AddFace(ctx, false, proto.Image{
			ID:   proto.FeatureID("id1"),
			URI:  img1,
			Tag:  "wawa1",
			Desc: json.RawMessage("desc1"),
			BoundingBox: proto.BoundingBox{
				Pts: [][2]int{[2]int{108, 55}, [2]int{391, 55}, [2]int{391, 357}, [2]int{108, 357}},
			},
		})
		a.Nil(err)

		data := make([]proto.Data, 1)
		data[0].URI = img1
		data[0].Attribute.BoundingBoxes = append(data[0].Attribute.BoundingBoxes, proto.BoundingBox{
			Pts: [][2]int{[2]int{108, 55}, [2]int{391, 55}, [2]int{391, 357}, [2]int{108, 357}},
		})
		fvss, faceBoxess, err := s.DetectAndFetchFeature(ctx, false, data)
		a.Nil(err)
		a.Len(fvss, 1)
		a.Len(faceBoxess, 1)
		a.Len(fvss[0], 1)
		a.Len(faceBoxess[0], 1)

		ss, err := g.SearchFace(ctx, 0.99, 100, fvss)
		a.Nil(err)
		a.Len(ss, 1)
		a.Len(ss[0], 1)
		a.Len(ss[0][0].Faces, 1)
		a.Equal(ss[0][0].Faces[0].ID, proto.FeatureID("id1"))
	})
}

type mockFaceFeature struct {
	FacesQuality []proto.FaceDetectBox
	Features     proto.FeatureValue
	Err          error
}

func (m mockFaceFeature) Face(ctx context.Context, img proto.ImageURI, pts [][2]int) (proto.FeatureValue, error) {
	// onlt match the first pts
	if pts[0][0] != m.FacesQuality[0].BoundingBox.Pts[0][0] ||
		pts[0][1] != m.FacesQuality[0].BoundingBox.Pts[0][1] ||
		pts[1][0] != m.FacesQuality[0].BoundingBox.Pts[1][0] ||
		pts[1][1] != m.FacesQuality[0].BoundingBox.Pts[1][1] ||
		pts[2][0] != m.FacesQuality[0].BoundingBox.Pts[2][0] ||
		pts[2][1] != m.FacesQuality[0].BoundingBox.Pts[2][1] ||
		pts[3][0] != m.FacesQuality[0].BoundingBox.Pts[3][0] ||
		pts[3][1] != m.FacesQuality[0].BoundingBox.Pts[3][1] {
		return nil, errors.New("mismatch pts")
	}
	return m.Features, m.Err
}

func (m mockFaceFeature) FaceBoxes(context.Context, proto.ImageURI) ([]proto.BoundingBox, error) {
	faces := make([]proto.BoundingBox, 0)
	for _, fq := range m.FacesQuality {
		faces = append(faces, fq.BoundingBox)
	}
	return faces, m.Err
}

func (m mockFaceFeature) FaceBoxesQuality(context.Context, proto.ImageURI) ([]proto.FaceDetectBox, error) {
	return m.FacesQuality, m.Err
}

func TestParseImageFeatures(t *testing.T) {
	t.Skip()
	var (
		ctx      = context.Background()
		features []proto.Feature
		err      error
		pts      [][2]int
	)

	fg := &FaceGroup{
		baseGroup: &_BaseGroup{
			Name: "test",
		},
		manager: &FaceGroups{
			config: FaceGroupsConfig{
				MultiFacesMode: 0,
			},
		},
	}

	// one face, 51*51
	pts = [][2]int{[2]int{0, 0}, [2]int{51, 0}, [2]int{51, 51}, [2]int{0, 51}}
	fg.manager.facefeature = mockFaceFeature{
		FacesQuality: []proto.FaceDetectBox{proto.FaceDetectBox{
			BoundingBox: proto.BoundingBox{Pts: pts, Score: 0.99},
		}},
	}
	features, _, err = fg.parseImageFeatures(ctx, false, [2]int{50, 50}, proto.Image{})
	assert.Nil(t, err)
	assert.Equal(t, 0, len(features[0].Value))

	// one face, 49*50
	pts = [][2]int{[2]int{0, 0}, [2]int{49, 0}, [2]int{49, 50}, [2]int{0, 50}}
	fg.manager.facefeature = mockFaceFeature{
		FacesQuality: []proto.FaceDetectBox{proto.FaceDetectBox{
			BoundingBox: proto.BoundingBox{Pts: pts, Score: 0.99},
		}},
	}
	_, _, err = fg.parseImageFeatures(ctx, false, [2]int{50, 50}, proto.Image{})
	assert.Equal(t, "No face found in image", err.Error())

	//choose Pts face
	pts = [][2]int{[2]int{108, 55}, [2]int{391, 55}, [2]int{391, 357}, [2]int{108, 357}}
	fg.manager.facefeature = mockFaceFeature{
		FacesQuality: []proto.FaceDetectBox{proto.FaceDetectBox{
			BoundingBox: proto.BoundingBox{Pts: pts, Score: 0.99},
		}},
	}
	features, _, err = fg.parseImageFeatures(ctx, false, [2]int{50, 50}, proto.Image{
		BoundingBox: proto.BoundingBox{
			Pts: [][2]int{[2]int{108, 55}, [2]int{391, 55}, [2]int{391, 357}, [2]int{108, 357}},
		},
	})
	assert.Nil(t, err)
	assert.Equal(t, 0, len(features[0].Value))

	// two faces, add_multi_face = false
	fg.manager.facefeature = mockFaceFeature{
		FacesQuality: []proto.FaceDetectBox{
			proto.FaceDetectBox{BoundingBox: proto.BoundingBox{Pts: [][2]int{[2]int{0, 0}, [2]int{52, 0}, [2]int{52, 52}, [2]int{0, 52}}, Score: 0.99}},
			proto.FaceDetectBox{BoundingBox: proto.BoundingBox{Pts: [][2]int{[2]int{0, 0}, [2]int{51, 0}, [2]int{51, 51}, [2]int{0, 51}}, Score: 0.99}},
		},
	}
	_, _, err = fg.parseImageFeatures(ctx, false, [2]int{50, 50}, proto.Image{})
	assert.Equal(t, "Multiple faces found in image", err.Error())

	// two faces, add_multi_face = true
	fg.manager.config.MultiFacesMode = 1
	features, _, err = fg.parseImageFeatures(ctx, false, [2]int{50, 50}, proto.Image{ID: "test"})
	assert.Nil(t, err)
	assert.Equal(t, 0, len(features[0].Value))

	return
}

func TestFaceQuality(t *testing.T) {
	t.Skip()
	var (
		ctx      = context.Background()
		err      error
		pts      [][2]int
		ff       mockFaceFeature
		features []proto.Feature
	)

	fg := &FaceGroup{
		baseGroup: &_BaseGroup{
			Name: "test",
		},
		manager: &FaceGroups{
			config: FaceGroupsConfig{
				MultiFacesMode: 0,
			},
		},
	}
	pts = [][2]int{[2]int{0, 0}, [2]int{51, 0}, [2]int{51, 51}, [2]int{0, 51}}
	ff.FacesQuality = append(ff.FacesQuality, proto.FaceDetectBox{
		BoundingBox: proto.BoundingBox{Pts: pts, Score: 0.99},
		Quality:     proto.FaceQuality{Quality: proto.FaceQualityClear, Orientation: proto.FaceOrientationUp},
	})
	fg.manager.facefeature = ff

	features, _, err = fg.parseImageFeatures(ctx, false, [2]int{50, 50}, proto.Image{})
	assert.Nil(t, err)
	assert.Equal(t, len(features), 1)
	assert.Equal(t, features[0].FaceQuality.Quality, proto.FaceQualityClear)
	assert.Equal(t, features[0].FaceQuality.Orientation, proto.FaceOrientationUp)
	assert.Nil(t, features[0].FaceQuality.QualityScore)

	features, _, err = fg.parseImageFeatures(ctx, true, [2]int{50, 50}, proto.Image{})
	assert.Nil(t, err)
	assert.Equal(t, len(features), 1)
	assert.Equal(t, features[0].FaceQuality.Quality, proto.FaceQualityClear)
	assert.Equal(t, features[0].FaceQuality.Orientation, proto.FaceOrientationUp)
	assert.Nil(t, features[0].FaceQuality.QualityScore)

	ff.FacesQuality[0].Quality.Quality = proto.FaceQualityBlur
	ff.FacesQuality[0].Quality.QualityScore = map[proto.FaceQualityClass]float32{proto.FaceQualityBlur: 0.99, proto.FaceQualityClear: 0.01}
	fg.manager.facefeature = ff
	features, _, err = fg.parseImageFeatures(ctx, false, [2]int{50, 50}, proto.Image{})
	assert.Nil(t, err)
	assert.Equal(t, len(features), 1)
	assert.Equal(t, features[0].FaceQuality.Quality, proto.FaceQualityBlur)
	assert.Equal(t, features[0].FaceQuality.Orientation, proto.FaceOrientationUp)
	assert.Equal(t, len(features[0].FaceQuality.QualityScore), 2)
	assert.Equal(t, features[0].FaceQuality.QualityScore[proto.FaceQualityBlur], float32(0.99))
	assert.Equal(t, features[0].FaceQuality.QualityScore[proto.FaceQualityClear], float32(0.01))

	_, _, err = fg.parseImageFeatures(ctx, true, [2]int{50, 50}, proto.Image{})
	assert.Equal(t, err, ErrBlurFace)

	ff.FacesQuality[0].Quality.Quality = proto.FaceQualityCover
	fg.manager.facefeature = ff
	_, _, err = fg.parseImageFeatures(ctx, true, [2]int{50, 50}, proto.Image{})
	assert.Equal(t, err, ErrCoverFace)

	ff.FacesQuality[0].Quality.Quality = proto.FaceQualityPose
	fg.manager.facefeature = ff
	_, _, err = fg.parseImageFeatures(ctx, true, [2]int{50, 50}, proto.Image{})
	assert.Equal(t, err, ErrPoseFace)

	ff.FacesQuality[0].Quality.Quality = proto.FaceQualityClear
	ff.FacesQuality[0].Quality.Orientation = proto.FaceOrientationDown
	fg.manager.facefeature = ff
	_, _, err = fg.parseImageFeatures(ctx, true, [2]int{50, 50}, proto.Image{})
	assert.Equal(t, err, ErrOrientationNotUp)
}
