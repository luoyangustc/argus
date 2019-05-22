package service

import (
	"context"
	"encoding/base64"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"hash/crc32"
	"io/ioutil"
	"math"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"testing"
	"time"

	"gopkg.in/mgo.v2"
	"gopkg.in/mgo.v2/bson"

	mgoutil "github.com/qiniu/db/mgoutil.v3"
	restrpc "github.com/qiniu/http/restrpc.v1"
	servestk "github.com/qiniu/http/servestk.v1"
	"github.com/stretchr/testify/assert"

	"qiniu.com/argus/com/uri"
	feature_group "qiniu.com/argus/feature_group_private"
	"qiniu.com/argus/feature_group_private/feature"
	"qiniu.com/argus/feature_group_private/proto"
	"qiniu.com/argus/feature_group_private/search"
	tusosearch "qiniu.com/argus/tuso/search"
)

var ImageFeature feature.ImageFeature

const (
	imageFeatureSize     = 2048 * 4
	faceFeatureSize      = 512 * 4
	defaultClusterPrefix = "v1/cluster"
	defaultGroup         = "cluster_group"
)

type mockServer struct {
	NoFaceBase64URI string
}

func createFeatureValues(seed string, size int) (fvs []byte) {
	runes := []rune(seed)
	var offset = size
	if len(runes) < offset {
		offset = len(runes)
	}
	for j := 1; j < size+1; j++ {
		bs := make([]byte, 4)
		binary.LittleEndian.PutUint32(bs, math.Float32bits(float32(runes[j%offset])))
		fvs = append(fvs, bs...)
	}
	return
}

func littleEndianToBigEndian(a []byte) []byte {
	b := make([]byte, len(a))
	for i := 0; i < len(a); i += 4 {
		binary.BigEndian.PutUint32(b[i:i+4], binary.LittleEndian.Uint32(a[i:i+4]))
	}
	return b
}

func (s *mockServer) HandleImageFeature(w http.ResponseWriter, r *http.Request) {
	body, _ := ioutil.ReadAll(r.Body)
	defer r.Body.Close()
	fv := createFeatureValues(string(body), imageFeatureSize/4)
	w.Write(fv)
}

func (s *mockServer) HandleFaceFeature(w http.ResponseWriter, r *http.Request) {
	body, _ := ioutil.ReadAll(r.Body)
	defer r.Body.Close()
	fv := createFeatureValues(string(body), faceFeatureSize/4)
	tusosearch.NormFeatures(fv, faceFeatureSize)
	fv = littleEndianToBigEndian(fv)
	w.Write(fv)
}

func (s *mockServer) HandleFaceDetectV3(w http.ResponseWriter, r *http.Request) {
	type detection struct {
		Index        int                    `json:"index"`
		Class        string                 `json:"class"`
		Score        float32                `json:"score"`
		Pts          [][2]int               `json:"pts"`
		Orientation  proto.FaceOrientation  `json:"orientation,omitempty"`
		Quality      proto.FaceQualityClass `json:"quality,omitempty"`
		QualityScore map[string]float32     `json:"q_score,omitempty"`
	}
	type result struct {
		Detections []detection `json:"detections"`
	}
	type resp struct {
		Code    int    `json:"code"`
		Message string `json:"message"`
		Result  result `json:"result"`
	}
	ds := []detection{detection{
		Pts: [][2]int{[2]int{108, 55}, [2]int{391, 55}, [2]int{391, 357}, [2]int{108, 357}},
	}}
	type request struct {
		Data struct {
			URI string `json:"uri"`
		} `json:"data"`
		Params struct {
			UseQuality int `json:"use_quality"`
		} `json:"params"`
	}
	var req request
	body, _ := ioutil.ReadAll(r.Body)
	if err := json.Unmarshal(body, &req); err != nil {
		w.WriteHeader(http.StatusBadRequest)
		return
	}
	if req.Params.UseQuality == 1 {
		ds[0].Quality = proto.FaceQualityClear
		ds[0].Orientation = proto.FaceOrientationUp
		ds[0].QualityScore = map[string]float32{"clear": 0.99999, "blur": 0.00001}
	}
	if strings.Contains(req.Data.URI, "https://odum9helk.qnssl.com/FjkrNUuQ8bTqaPAEsgGZBAYMi5qS") ||
		strings.Contains(req.Data.URI, s.NoFaceBase64URI) {
		ds = []detection{}
	}
	v, _ := json.Marshal(resp{Result: result{Detections: ds}})
	w.Write(v)
}

func (s *mockServer) HandleCreateGroup(w http.ResponseWriter, r *http.Request) {
	return
}

func (s *mockServer) HandleRemoveGroup(w http.ResponseWriter, r *http.Request) {
	return
}
func (s *mockServer) HandleAddFeature(w http.ResponseWriter, r *http.Request) {
	return
}
func (s *mockServer) HandleDeleteFeature(w http.ResponseWriter, r *http.Request) {
	return
}
func (s *mockServer) HandleUpdateFeature(w http.ResponseWriter, r *http.Request) {
	return
}

func (s *mockServer) HandleSearchFeature(w http.ResponseWriter, r *http.Request) {
	var ret [][]feature_group.FeatureSearchItem
	v, _ := json.Marshal(ret)
	w.Write(v)
	return
}

func base64ImageURI(image string) string {
	cli := uri.New(uri.WithHTTPHandler())
	resp, err := cli.Get(context.Background(), uri.Request{URI: string(image)})
	if err != nil {
		return ""
	}
	defer resp.Body.Close()
	buf, err := ioutil.ReadAll(resp.Body)
	if err != nil || len(buf) == 0 {
		return ""
	}
	return "data:application/octet-stream;base64," + base64.StdEncoding.EncodeToString(buf)
}

func runMockServer() string {
	svr := &mockServer{
		NoFaceBase64URI: base64ImageURI("https://odum9helk.qnssl.com/FjkrNUuQ8bTqaPAEsgGZBAYMi5qS"),
	}
	mux := servestk.New(restrpc.DefaultServeMux)
	//mux := http.NewServeMux()
	mux.HandleFunc("POST /v1/eval/image-feature", svr.HandleImageFeature)
	mux.HandleFunc("POST /v1/eval/facex-feature-v4", svr.HandleFaceFeature)
	mux.HandleFunc("POST /v1/eval/facex-detect", svr.HandleFaceDetectV3)
	mux.HandleFunc("POST /"+defaultClusterPrefix+"/groups/"+defaultGroup, svr.HandleCreateGroup)
	mux.HandleFunc("POST /"+defaultClusterPrefix+"/groups/"+defaultGroup+"/remove", svr.HandleRemoveGroup)
	mux.HandleFunc("POST /"+defaultClusterPrefix+"/groups/"+defaultGroup+"/feature/add", svr.HandleAddFeature)
	mux.HandleFunc("POST /"+defaultClusterPrefix+"/groups/"+defaultGroup+"/delete", svr.HandleDeleteFeature)
	mux.HandleFunc("POST /"+defaultClusterPrefix+"/groups/"+defaultGroup+"/feature/update", svr.HandleUpdateFeature)
	mux.HandleFunc("POST /"+defaultClusterPrefix+"/groups/"+defaultGroup+"/feature/search", svr.HandleSearchFeature)
	ts := httptest.NewServer(mux)
	return ts.URL
}

func TestBaseGroups(t *testing.T) {
	var featureHost = runMockServer()
	if os.Getenv("NOMOCK") != "" {
		featureHost = "http://ava-serving-gate.cs.cg.dora-internal.qiniu.io:5001"
	}
	a := assert.New(t)
	ctx := context.Background()
	ImageFeature = feature.NewImageFeature(
		featureHost,
		time.Duration(15)*time.Second,
		2048*4,
	)
	var (
		MGO_HOST = "mongodb://127.0.0.1"
		MGO_DB   = "feature_group_private_test_01"
	)

	sess, err := mgo.Dial(MGO_HOST)
	a.Nil(err)
	_ = sess.DB(MGO_DB).DropDatabase()
	baseConfig := BaseGroupsConfig{
		MgoConfig: mgoutil.Config{
			Host: MGO_HOST,
			DB:   MGO_DB,
			Mode: "strong",
		},
		CollSessionPoolLimit: 50,
		Sets: search.Config{
			Dimension: 2048,
			Precision: 4,
			Version:   0,
			DeviceID:  0,
			BlockSize: 2048 * 4 * 10,
			BlockNum:  100,
			BatchSize: 5,
		},
	}

	s, err := NewBaseGroups(ctx, baseConfig, "")
	a.Nil(err)

	t.Run("插入一个Feature可以搜索出来", func(t *testing.T) {
		capacity := 100
		groupName := proto.GroupName("group1")
		err = s.New(ctx, false, groupName, proto.GroupConfig{
			Capacity: capacity,
		})
		a.Nil(err)
		g, err := s.Get(ctx, groupName)
		a.Nil(err)
		gs, err := s.All(ctx)
		a.Nil(err)
		a.Len(gs, 1)
		a.Equal(gs[0], proto.GroupName(groupName))
		n, err := g.Count(ctx)
		a.Nil(err)
		a.Equal(0, n)
		feature1, err := getImgFeature("http://q.hi-hi.cn/1.png")
		a.Nil(err)
		a.Nil(g.Add(ctx, false, proto.Feature{
			ID:    proto.FeatureID("id1"),
			Value: feature1,
			Tag:   "tag1",
			Desc:  json.RawMessage(`"aaa"`),
		}))
		tags, nextMarker, err := g.Tags(ctx, "", 10)
		if a.Nil(err) {
			a.Len(tags, 1)
			a.Empty(nextMarker)
		}
		count, err := g.CountTags(ctx)
		if a.Nil(err) {
			a.Equal(1, count)
		}
		config := g.Config(ctx)
		a.Equal(capacity, config.Capacity)
		ss, err := g.Search(ctx, false, 0.99, 100, feature1)
		if a.Nil(err) {
			a.Len(ss, 1)
			a.Len(ss[0], 1)
			a.Equal(ss[0][0].Value.ID, proto.FeatureID("id1"))
			a.Equal(ss[0][0].Value.Tag, proto.FeatureTag("tag1"))
			a.Equal(ss[0][0].Value.Desc, json.RawMessage(`"aaa"`))
			a.Equal(ss[0][0].Value.Value, feature1)
		}
	})

	t.Run("插入图片后可以用label过滤出", func(t *testing.T) {
		groupName := proto.GroupName("group2")
		err = s.New(ctx, false, groupName, proto.GroupConfig{
			Capacity: 100,
		})
		a.Nil(err)
		g, err := s.Get(ctx, groupName)
		a.Nil(err)
		feature1, err := getImgFeature("http://q.hi-hi.cn/1.png")
		a.Nil(err)
		a.Nil(g.Add(ctx, false, proto.Feature{
			ID:    proto.FeatureID("id1"),
			Value: feature1,
			Tag:   "tag1",
			Desc:  json.RawMessage(`"desc1"`),
		}))
		a.Nil(g.Add(ctx, false, proto.Feature{
			ID:    proto.FeatureID("id2"),
			Value: feature1,
			Tag:   "tag2",
			Desc:  json.RawMessage(`"desc2"`),
		}))
		tags, nextMarker, err := g.Tags(ctx, "", 10)
		if a.Nil(err) {
			a.Len(tags, 2)
			a.Empty(nextMarker)
		}
		ss, _, err := g.FilterByTag(ctx, "tag2", "", 10)
		a.Nil(err)
		a.Len(ss, 1)
		a.Equal(ss[0].ID, proto.FeatureID("id2"))
		a.Equal(ss[0].Tag, proto.FeatureTag("tag2"))
		a.Equal(ss[0].Desc, json.RawMessage(`"desc2"`))
		a.Equal(ss[0].Value, feature1)
	})

	t.Run("Destroy", func(t *testing.T) {
		groupName := proto.GroupName("group3")
		err = s.New(ctx, false, groupName, proto.GroupConfig{
			Capacity: 100,
		})
		a.Nil(err)
		g, err := s.Get(ctx, groupName)
		a.Nil(err)
		feature1, err := getImgFeature("http://q.hi-hi.cn/1.png")
		a.Nil(err)
		a.Nil(g.Add(ctx, false, proto.Feature{
			ID:    proto.FeatureID("id1"),
			Value: feature1,
			Tag:   "tag1",
			Desc:  json.RawMessage(`"desc1"`),
		}))
		a.Nil(g.Add(ctx, false, proto.Feature{
			ID:    proto.FeatureID("id2"),
			Value: feature1,
			Tag:   "tag2",
			Desc:  json.RawMessage(`"desc2"`),
		}))
		err = g.Destroy(ctx, false)
		a.Nil(err)
		_, err = s.Get(ctx, "group4")
		a.NotNil(err)
	})

	t.Run("CRUD多张图片", func(t *testing.T) {
		groupName := proto.GroupName("group4")
		err = s.New(ctx, false, groupName, proto.GroupConfig{
			Capacity: 100,
		})
		a.Nil(err)
		g, err := s.Get(ctx, groupName)
		a.Nil(err)
		feature1, err := getImgFeature("http://q.hi-hi.cn/1.png")
		a.Nil(err)
		feature2, err := getImgFeature("https://www.qiniu.com/assets/icon-controllale@2x-47c22ae3192d5b1a26f8ccb4852d67ea8a1d10d5ab357bda51959edabdab1237.png")
		a.Nil(err)
		feature3, err := getImgFeature("https://odum9helk.qnssl.com/FjkrNUuQ8bTqaPAEsgGZBAYMi5qS")
		a.Nil(err)
		a.Nil(g.Add(ctx, false, proto.Feature{
			ID:    proto.FeatureID("id1"),
			Value: feature1,
			Tag:   "tag1",
			Desc:  json.RawMessage(`"desc1"`),
		}, proto.Feature{
			ID:    proto.FeatureID("id2"),
			Value: feature2,
			Tag:   "tag2",
			Desc:  json.RawMessage(`"desc2"`),
		}, proto.Feature{
			ID:    proto.FeatureID("id3"),
			Value: feature3,
			Tag:   "tag3",
			Desc:  json.RawMessage(`"desc3"`),
		}, proto.Feature{
			ID:    proto.FeatureID("id4"),
			Value: feature3,
			Tag:   "tag3",
			Desc:  json.RawMessage(`"desc4"`),
		}))

		n, err := g.Count(ctx)
		a.Nil(err)
		a.Equal(n, 4)
		tags, nextMarker, err := g.Tags(ctx, "", 10)
		if a.Nil(err) {
			a.Len(tags, 3)
			a.Empty(nextMarker)
		}
		ids, err := g.Delete(ctx, false, "id1")
		a.Nil(err)
		a.Equal([]proto.FeatureID{"id1"}, ids)
		ids, err = g.Delete(ctx, false, "id1")
		a.Nil(err)
		a.Equal([]proto.FeatureID{}, ids)
		tags, nextMarker, err = g.Tags(ctx, "", 10)
		if a.Nil(err) {
			a.Len(tags, 2)
			a.Empty(nextMarker)
		}
		n, err = g.Count(ctx)
		a.Nil(err)
		a.Equal(n, 3)

		ss, err := g.Search(ctx, false, 0.99, 100, feature1)
		a.Nil(err)
		a.Len(ss[0], 0)

		ss, err = g.Search(ctx, false, 0.99, 100, feature2)
		a.Nil(err)
		a.Len(ss[0], 1)
		a.Equal(ss[0][0].Value.ID, proto.FeatureID("id2"))
		a.Equal(ss[0][0].Value.Tag, proto.FeatureTag("tag2"))
		a.Equal(ss[0][0].Value.Desc, json.RawMessage(`"desc2"`))
		a.Equal(ss[0][0].Value.Value, feature2)

		ss, err = g.Search(ctx, false, 0.99, 100, feature3)
		a.Nil(err)
		a.Len(ss[0], 2)
	})
}

func getImgFeature(url string) (proto.FeatureValue, error) {
	ctx := context.Background()
	buf := proto.ImageURI(url)
	fv, err := ImageFeature.Image(ctx, buf)
	if err != nil {
		return nil, err
	}
	return fv, nil
}

func TektClusterBaseGroup(t *testing.T) {
	var featureHost = runMockServer()
	if os.Getenv("NOMOCK") != "" {
		featureHost = "http://ava-serving-gate.cs.cg.dora-internal.qiniu.io:5001"
	}
	ImageFeature = feature.NewImageFeature(
		featureHost,
		time.Duration(15)*time.Second,
		2048*4,
	)
	var (
		a        = assert.New(t)
		MGO_HOST = "mongodb://127.0.0.1"
		MGO_DB   = search.GetRandomString(10)
		s        *BaseGroups
		ctx      = context.Background()
	)

	sess, err := mgo.Dial(MGO_HOST)
	a.Nil(err)
	defer sess.DB(MGO_DB).DropDatabase()
	baseConfig := BaseGroupsConfig{
		MgoConfig: mgoutil.Config{
			Host: MGO_HOST,
			DB:   MGO_DB,
			Mode: "strong",
		},
		ClusterMode:          true,
		ClusterSize:          2,
		Address:              "126.0.0.1:6198",
		BaseFeatureTimeout:   10,
		CollSessionPoolLimit: 50,
		Sets: search.Config{
			Dimension: 2048,
			Precision: 4,
			Version:   0,
			DeviceID:  0,
			BlockSize: 2048 * 4 * 10,
			BlockNum:  10,
			BatchSize: 5,
		},
	}

	groupConfig := proto.GroupConfig{
		Dimension: 2048,
		Precision: 4,
		Capacity:  0,
		Version:   0,
	}
	a.Nil(sess.DB(MGO_DB).C("groups").Insert(bson.M{"group_config": groupConfig, "name": proto.GroupName(defaultGroup)}))
	fv := createFeatureValues("first image", 2048/4)
	tusosearch.NormFeatures(fv, faceFeatureSize)
	fv = littleEndianToBigEndian(fv)
	a.Nil(sess.DB(MGO_DB).C("features").Insert(bson.M{"id": "feature001", "value": fv, "group": proto.GroupName(defaultGroup), "hash_key": proto.FeatureHashKey(1)}))
	a.Nil(sess.DB(MGO_DB).C("features").Insert(bson.M{"id": "feature002", "value": fv, "group": proto.GroupName(defaultGroup)}))

	t.Run("集群初始化", func(t *testing.T) {
		node := proto.Node{
			Address:  proto.NodeAddress(strings.TrimPrefix(featureHost, "http://")),
			Capacity: proto.NodeCapacity(0),
			State:    proto.NodeStateInitializing,
		}
		a.Nil(sess.DB(MGO_DB).C("nodes").Insert(node))
		go func() {
			time.Sleep(1 * time.Second)
			node.State = proto.NodeStateReady
			a.Nil(sess.DB(MGO_DB).C("nodes").Update(bson.M{"address": proto.NodeAddress(strings.TrimPrefix(featureHost, "http://"))}, node))
		}()
		s, err = NewBaseGroups(ctx, baseConfig, defaultClusterPrefix)
		a.Nil(err)
		a.Nil(sess.DB(MGO_DB).C("nodes").Find(bson.M{"address": "126.0.0.1:6198"}).One(&node))
		a.Equal(proto.NodeCapacity(2048*10*4*100-1), node.Capacity)
		a.Equal(proto.NodeStateReady, node.State)
		g, err := s.Get(ctx, proto.GroupName(defaultGroup))
		a.Nil(err)
		a.Nil(g.Destroy(ctx, false))
	})

	t.Run("插入一个Feature可以搜索出来", func(t *testing.T) {
		groupName := proto.GroupName(defaultGroup)

		err := s.New(ctx, false, groupName, proto.GroupConfig{
			Capacity: 100,
		})
		a.Nil(err)

		g, err := s.Get(ctx, groupName)
		a.Nil(err)
		gs, err := s.All(ctx)
		a.Nil(err)
		a.Len(gs, 1)
		a.Equal(gs[0], proto.GroupName(groupName))
		n, err := g.Count(ctx)
		a.Nil(err)
		a.Equal(0, n)
		feature1, err := getImgFeature("http://q.hi-hi.cn/1.png")
		a.Nil(err)
		a.Nil(g.Add(ctx, false, proto.Feature{
			ID:      proto.FeatureID("id1"),
			Value:   feature1,
			Tag:     "tag1",
			Desc:    json.RawMessage(`"aaa"`),
			HashKey: proto.FeatureHashKey(1),
		}))
		tags, nextMarker, err := g.Tags(ctx, "", 10)
		if a.Nil(err) {
			a.Len(tags, 1)
			a.Empty(nextMarker)
		}
		ss, err := g.Search(ctx, false, 0.99, 100, feature1)
		a.Nil(err)
		a.Len(ss, 1)
		a.Len(ss[0], 1)
		a.Equal(ss[0][0].Value.ID, proto.FeatureID("id1"))
		a.Equal(ss[0][0].Value.Tag, proto.FeatureTag("tag1"))
		a.Equal(ss[0][0].Value.Desc, json.RawMessage(`"aaa"`))
		a.Equal(ss[0][0].Value.Value, feature1)

		g, err = s.Get(ctx, groupName)
		a.Nil(err)
		a.Nil(g.Destroy(ctx, false))
	})

	t.Run("插入图片后可以用label过滤出", func(t *testing.T) {
		groupName := proto.GroupName(defaultGroup)
		err = s.New(ctx, false, groupName, proto.GroupConfig{
			Capacity: 100,
		})
		a.Nil(err)
		g, err := s.Get(ctx, groupName)
		a.Nil(err)
		feature1, err := getImgFeature("http://q.hi-hi.cn/1.png")
		a.Nil(err)
		a.Nil(g.Add(ctx, false, proto.Feature{
			ID:      proto.FeatureID("id1"),
			Value:   feature1,
			Tag:     "tag1",
			Desc:    json.RawMessage(`"desc1"`),
			HashKey: proto.FeatureHashKey(1),
		}))
		a.Nil(g.Add(ctx, false, proto.Feature{
			ID:      proto.FeatureID("id2"),
			Value:   feature1,
			Tag:     "tag2",
			Desc:    json.RawMessage(`"desc2"`),
			HashKey: proto.FeatureHashKey(2),
		}))
		tags, nextMarker, err := g.Tags(ctx, "", 10)
		if a.Nil(err) {
			a.Len(tags, 2)
			a.Empty(nextMarker)
		}
		ss, _, err := g.FilterByTag(ctx, "tag2", "", 10)
		a.Nil(err)
		a.Len(ss, 1)
		a.Equal(ss[0].ID, proto.FeatureID("id2"))
		a.Equal(ss[0].Tag, proto.FeatureTag("tag2"))
		a.Equal(ss[0].Desc, json.RawMessage(`"desc2"`))
		a.Equal(ss[0].Value, feature1)
		g, err = s.Get(ctx, groupName)
		a.Nil(err)
		a.Nil(g.Destroy(ctx, false))
	})

	t.Run("更新图片", func(t *testing.T) {
		groupName := proto.GroupName(defaultGroup)
		err = s.New(ctx, false, groupName, proto.GroupConfig{
			Capacity: 100,
		})
		a.Nil(err)
		g, err := s.Get(ctx, groupName)
		a.Nil(err)
		feature1, err := getImgFeature("http://q.hi-hi.cn/1.png")
		a.Nil(err)
		a.Nil(g.Add(ctx, false, proto.Feature{
			ID:      proto.FeatureID("id1"),
			Value:   feature1,
			Tag:     "tag1",
			Desc:    json.RawMessage(`"desc1"`),
			HashKey: proto.FeatureHashKey(1),
		}))
		a.Nil(g.Add(ctx, false, proto.Feature{
			ID:      proto.FeatureID("id2"),
			Value:   feature1,
			Tag:     "tag2",
			Desc:    json.RawMessage(`"desc2"`),
			HashKey: proto.FeatureHashKey(2),
		}))
		err = g.Destroy(ctx, false)
		a.Nil(err)
		_, err = s.Get(ctx, "group4")
		a.NotNil(err)
	})

	t.Run("Destroy", func(t *testing.T) {
		groupName := proto.GroupName(defaultGroup)
		err = s.New(ctx, false, groupName, proto.GroupConfig{
			Capacity: 100,
		})
		a.Nil(err)
		g, err := s.Get(ctx, groupName)
		a.Nil(err)
		feature1, err := getImgFeature("http://q.hi-hi.cn/1.png")
		a.Nil(err)
		a.Nil(g.Add(ctx, false, proto.Feature{
			ID:      proto.FeatureID("id1"),
			Value:   feature1,
			Tag:     "tag1",
			Desc:    json.RawMessage(`"desc1"`),
			HashKey: proto.FeatureHashKey(1),
		}))
		tags, nextMarker, err := g.Tags(ctx, "", 10)
		if a.Nil(err) {
			a.Len(tags, 1)
			a.Empty(nextMarker)
			a.Equal(proto.FeatureTag("tag1"), tags[0].Name)
		}
		a.Nil(g.Update(ctx, false, proto.Feature{
			ID:      proto.FeatureID("id1"),
			Value:   feature1,
			Tag:     "tag2",
			Desc:    json.RawMessage(`"desc2"`),
			HashKey: proto.FeatureHashKey(1),
		}))
		tags, nextMarker, err = g.Tags(ctx, "", 10)
		if a.Nil(err) {
			a.Len(tags, 1)
			a.Empty(nextMarker)
			a.Equal(proto.FeatureTag("tag2"), tags[0].Name)
		}
		err = g.Destroy(ctx, false)
		a.Nil(err)
	})

	t.Run("CRUD多张图片", func(t *testing.T) {
		groupName := proto.GroupName(defaultGroup)
		err = s.New(ctx, false, groupName, proto.GroupConfig{
			Capacity: 100,
		})
		a.Nil(err)
		g, err := s.Get(ctx, groupName)
		a.Nil(err)
		feature1, err := getImgFeature("http://q.hi-hi.cn/1.png")
		a.Nil(err)
		feature2, err := getImgFeature("https://www.qiniu.com/assets/icon-controllale@2x-47c22ae3192d5b1a26f8ccb4852d67ea8a1d10d5ab357bda51959edabdab1237.png")
		a.Nil(err)
		feature3, err := getImgFeature("https://odum9helk.qnssl.com/FjkrNUuQ8bTqaPAEsgGZBAYMi5qS")
		a.Nil(err)
		a.Nil(g.Add(ctx, false, proto.Feature{
			ID:      proto.FeatureID("tA0qKR2hc17H"),
			Value:   feature1,
			Tag:     "tag1",
			Desc:    json.RawMessage(`"desc1"`),
			HashKey: proto.FeatureHashKey(crc32.ChecksumIEEE([]byte("tA0qKR2hc17H"))),
		}, proto.Feature{
			ID:      proto.FeatureID("id2"),
			Value:   feature2,
			Tag:     "tag2",
			Desc:    json.RawMessage(`"desc2"`),
			HashKey: proto.FeatureHashKey(2),
		}, proto.Feature{
			ID:      proto.FeatureID("id3"),
			Value:   feature3,
			Tag:     "tag3",
			Desc:    json.RawMessage(`"desc3"`),
			HashKey: proto.FeatureHashKey(3),
		}, proto.Feature{
			ID:      proto.FeatureID("id4"),
			Value:   feature3,
			Tag:     "tag3",
			Desc:    json.RawMessage(`"desc4"`),
			HashKey: proto.FeatureHashKey(4),
		}))

		n, err := g.Count(ctx)
		a.Nil(err)
		a.Equal(n, 4)
		tags, nextMarker, err := g.Tags(ctx, "", 10)
		if a.Nil(err) {
			a.Empty(nextMarker)
			a.Len(tags, 3)
		}
		ids, err := g.Delete(ctx, false, "tA0qKR2hc17H")
		a.Nil(err)
		a.Equal([]proto.FeatureID{"tA0qKR2hc17H"}, ids)
		ids, err = g.Delete(ctx, false, "tA0qKR2hc17H")
		a.Nil(err)
		a.Equal([]proto.FeatureID{}, ids)
		tags, nextMarker, err = g.Tags(ctx, "", 10)
		if a.Nil(err) {
			a.Empty(nextMarker)
			a.Len(tags, 2)
		}
		n, err = g.Count(ctx)
		a.Nil(err)
		a.Equal(n, 3)

		ss, err := g.Search(ctx, false, 0.8, 100, feature1)
		a.Nil(err)
		a.Len(ss[0], 0)

		ss, err = g.Search(ctx, false, 0.99, 100, feature2)
		a.Nil(err)
		a.Len(ss[0], 1)
		a.Equal(ss[0][0].Value.ID, proto.FeatureID("id2"))
		a.Equal(ss[0][0].Value.Tag, proto.FeatureTag("tag2"))
		a.Equal(ss[0][0].Value.Desc, json.RawMessage(`"desc2"`))
		a.Equal(ss[0][0].Value.Value, feature2)

		ss, err = g.Search(ctx, false, 0.99, 100, feature3)
		a.Nil(err)
		a.Len(ss[0], 2)
		g, err = s.Get(ctx, groupName)
		a.Nil(err)
		a.Nil(g.Destroy(ctx, false))
	})
}

func TestBaseGroupsCapacity(t *testing.T) {
	var (
		featureHost = runMockServer()
		MGO_HOST    = "mongodb://127.0.0.1"
		MGO_DB      = search.GetRandomString(10)
	)
	if os.Getenv("NOMOCK") != "" {
		featureHost = "http://ava-serving-gate.cs.cg.dora-internal.qiniu.io:5001"
	}
	ctx := context.Background()
	a := assert.New(t)

	ImageFeature = feature.NewImageFeature(
		featureHost,
		time.Duration(15)*time.Second,
		2048*4,
	)

	sess, err := mgo.Dial(MGO_HOST)
	a.Nil(err)

	baseConfig := BaseGroupsConfig{
		MgoConfig: mgoutil.Config{
			Host: MGO_HOST,
			DB:   MGO_DB,
			Mode: "strong",
		},
		CollSessionPoolLimit: 50,
		Sets: search.Config{
			Dimension: 2048,
			Precision: 4,
			Version:   0,
			DeviceID:  0,
			BlockSize: 2048 * 4 * 3,
			BlockNum:  3,
			BatchSize: 5,
		},
	}

	groupConfig := proto.GroupConfig{
		Dimension: 2048,
		Precision: 4,
		Capacity:  100,
		Version:   0,
	}

	feature, err := getImgFeature("http://q.hi-hi.cn/1.png")
	a.Nil(err)

	t.Run("单机单个Group插满", func(t *testing.T) {
		defer sess.DB(MGO_DB).DropDatabase()
		baseGroups, err := NewBaseGroups(ctx, baseConfig, "")
		groupName := proto.GroupName("group-full-1")
		err = baseGroups.New(ctx, false, groupName, groupConfig)
		a.Nil(err)
		g, err := baseGroups.Get(ctx, groupName)
		a.Nil(err)
		gs, err := baseGroups.All(ctx)
		a.Nil(err)
		a.Len(gs, 1)
		for i := 0; i < 9; i++ {
			a.Nil(g.Add(ctx, false, proto.Feature{
				ID:    proto.FeatureID(fmt.Sprintf("id-%d", i)),
				Value: feature,
				Tag:   "tag1",
				Desc:  json.RawMessage(`"aaa"`),
			}))
		}
		// 最后一个插入错误
		a.NotNil(g.Add(ctx, false, proto.Feature{
			ID:    proto.FeatureID("cannot-insert"),
			Value: feature,
			Tag:   "tag1",
			Desc:  json.RawMessage(`"aaa"`),
		}))
		// 额外的一个group插入错误
		groupName = proto.GroupName("group-full-extra")
		err = baseGroups.New(ctx, false, groupName, groupConfig)
		a.Nil(err)
		ng, err := baseGroups.Get(ctx, groupName)
		a.Nil(err)
		gs, err = baseGroups.All(ctx)
		a.Nil(err)
		a.Len(gs, 2)
		a.NotNil(ng.Add(ctx, false, proto.Feature{
			ID:    proto.FeatureID("cannot-insert"),
			Value: feature,
			Tag:   "tag1",
			Desc:  json.RawMessage(`"aaa"`),
		}))
		// 删除一个feature
		deleted, err := g.Delete(ctx, false, proto.FeatureID("id-0"))
		a.Nil(err)
		a.Len(deleted, 1)
		a.Equal(deleted[0], proto.FeatureID("id-0"))
		a.NotNil(ng.Add(ctx, false, proto.Feature{
			ID:    proto.FeatureID("cannot-insert"),
			Value: feature,
			Tag:   "tag1",
			Desc:  json.RawMessage(`"aaa"`),
		}))
		// 新的group中依然不能添加
		a.NotNil(ng.Add(ctx, false, proto.Feature{
			ID:    proto.FeatureID("cannot-insert"),
			Value: feature,
			Tag:   "tag1",
			Desc:  json.RawMessage(`"aaa"`),
		}))
		// 之前的group可以添加, 但只能添加一个
		a.Nil(g.Add(ctx, false, proto.Feature{
			ID:    proto.FeatureID("can-insert"),
			Value: feature,
			Tag:   "tag1",
			Desc:  json.RawMessage(`"aaa"`),
		}))
		a.NotNil(g.Add(ctx, false, proto.Feature{
			ID:    proto.FeatureID("cannot-insert"),
			Value: feature,
			Tag:   "tag1",
			Desc:  json.RawMessage(`"aaa"`),
		}))
	})

	t.Run("单机多个Group均可各自插满, 且无法继续插入", func(t *testing.T) {
		defer sess.DB(MGO_DB).DropDatabase()
		s, err := NewBaseGroups(ctx, baseConfig, "")
		a.Nil(err)
		for i := 0; i < 3; i++ {
			groupName := proto.GroupName(fmt.Sprintf("group-full-%d", i))
			err = s.New(ctx, false, groupName, groupConfig)
			a.Nil(err)
			g, err := s.Get(ctx, groupName)
			a.Nil(err)
			gs, err := s.All(ctx)
			a.Nil(err)
			a.Len(gs, i+1)
			for j := 0; j < 3; j++ {
				a.Nil(g.Add(ctx, false, proto.Feature{
					ID:    proto.FeatureID(fmt.Sprintf("id-%d-%d", i, j)),
					Value: feature,
					Tag:   "tag1",
					Desc:  json.RawMessage(`"aaa"`),
				}))
			}
		}
		// 之前的group也无法插入
		groupName := proto.GroupName(fmt.Sprintf("group-full-%d", 0))
		og, err := s.Get(ctx, groupName)
		a.Nil(err)
		a.NotNil(og.Add(ctx, false, proto.Feature{
			ID:    proto.FeatureID("cannot-insert"),
			Value: feature,
			Tag:   "tag1",
			Desc:  json.RawMessage(`"aaa"`),
		}))

		// 新申请的group也无法插入
		groupName = proto.GroupName("group-full-extra")
		err = s.New(ctx, false, groupName, groupConfig)
		a.Nil(err)
		ng, err := s.Get(ctx, groupName)
		a.Nil(err)
		a.NotNil(ng.Add(ctx, false, proto.Feature{
			ID:    proto.FeatureID("cannot-insert"),
			Value: feature,
			Tag:   "tag1",
			Desc:  json.RawMessage(`"aaa"`),
		}))

		// 删除一个feature
		deleted, err := og.Delete(ctx, false, proto.FeatureID("id-0-0"))
		a.Nil(err)
		a.Len(deleted, 1)
		a.Equal(deleted[0], proto.FeatureID("id-0-0"))

		// 新申请的group也无法插入
		a.NotNil(ng.Add(ctx, false, proto.Feature{
			ID:    proto.FeatureID("cannot-insert"),
			Value: feature,
			Tag:   "tag1",
			Desc:  json.RawMessage(`"aaa"`),
		}))

		// 其他的group也不能插入
		og1, err := s.Get(ctx, proto.GroupName(fmt.Sprintf("group-full-%d", 1)))
		a.Nil(err)
		a.NotNil(og1.Add(ctx, false, proto.Feature{
			ID:    proto.FeatureID("cannot-insert"),
			Value: feature,
			Tag:   "tag1",
			Desc:  json.RawMessage(`"aaa"`),
		}))
		// 之前的group可以添加, 但只能添加一个
		a.Nil(og.Add(ctx, false, proto.Feature{
			ID:    proto.FeatureID("can-insert"),
			Value: feature,
			Tag:   "tag1",
			Desc:  json.RawMessage(`"aaa"`),
		}))
		a.NotNil(og.Add(ctx, false, proto.Feature{
			ID:    proto.FeatureID("cannot-insert"),
			Value: feature,
			Tag:   "tag1",
			Desc:  json.RawMessage(`"aaa"`),
		}))
	})
}
