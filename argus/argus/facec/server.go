package facec

import (
	"context"
	"encoding/json"
	"net/http"
	"time"

	"github.com/qiniu/db/mgoutil.v3"
	"github.com/qiniu/xlog.v1"

	"qiniu.com/argus/argus/facec/client"
	"qiniu.com/argus/argus/facec/db"
	"qiniu.com/argus/argus/facec/imgprocess"
)

type Service struct {
	cl           cl
	dImage       db.ImageDao
	dFace        db.FaceDao
	dGroup       db.GroupDao
	dFeatureTask db.FeatureTaskDao
	dClusterTask db.ClusterTaskDao
	dAlias       db.AliasDao
	dVersion     db.DataVersionDao
	cfg          *Config
	groupMutex   GroupMutex
}

type Config struct {
	Hosts         client.Hosts   `json:"hosts"`
	RPCTimeoutMs  int            `json:"rpc_timeout_ms"`
	MgoConfig     mgoutil.Config `json:"mgo_config"`
	UseMock       bool           `json:"use_mock"`
	FeaturePrefix string         `json:"feature_prefix"`
}

func New(c Config) (s *Service, err error) {
	if err := db.Init(&c.MgoConfig); err != nil {
		return nil, err
	}
	imgDao, err := db.NewImageDao()
	if err != nil {
		return nil, err
	}
	faceDao, err := db.NewFaceDao()
	if err != nil {
		return nil, err
	}
	groupDao, err := db.NewGroupDao()
	if err != nil {
		return nil, err
	}
	featureTaskDao, err := db.NewFeatureTaskDao()
	if err != nil {
		return nil, err
	}
	clusterTaskDao, err := db.NewClusterTaskDao()
	if err != nil {
		return nil, err
	}
	aliasDao, err := db.NewAliasDao()
	if err != nil {
		return nil, err
	}

	dataVersionDao, err := db.NewDataVersionDao()
	if err != nil {
		return nil, err
	}

	s = &Service{
		cl: client.New(client.Config{
			Hosts:   c.Hosts,
			Timeout: time.Duration(c.RPCTimeoutMs) * time.Millisecond,
		}),
		dFeatureTask: featureTaskDao,

		groupMutex: NewGroupMutex(dataVersionDao),
	}
	s.dImage = imgDao
	s.dFace = faceDao
	s.dGroup = groupDao
	s.dClusterTask = clusterTaskDao
	s.dAlias = aliasDao
	s.dVersion = dataVersionDao
	if c.UseMock {
		s.cl = &mockCl{}
	}
	s.cfg = &c
	return
}

func (s *Service) getImgProcesser(urls []string) imgprocess.ImgProcesser {
	client := &http.Client{
		Timeout: time.Duration(s.cfg.RPCTimeoutMs) * time.Millisecond,
	}
	images := make([]imgprocess.Image, 0, len(urls))
	for _, url := range urls {
		images = append(images, imgprocess.NewThumbnailImage(url))
	}
	if s.cfg.UseMock {
		return imgprocess.NewMockProcess(images, nil)
	}
	return imgprocess.New(images, client)
}

type cl interface {
	PostFacexDex(ctx context.Context, args []string, env client.EvalEnv) (ret []client.FacexDetResp, err error)
	PostFacexFeature1(ctx context.Context, args client.FacexFeatureReq, env client.EvalEnv) (ret []byte, err error)
	//PostFacexFeature2(ctx context.Context, args client.FacexFeatureReq2) (ret client.FacexFeatureResp, err error)
	PostFacexCluster(ctx context.Context, args client.FacexClusterReq, env client.EvalEnv) (ret client.FacexClusterResp, err error)
}

type mockCl struct {
}

func (c *mockCl) PostFacexDex(ctx context.Context, args []string, env client.EvalEnv) (ret []client.FacexDetResp, err error) {
	result := `
[ 
	{ 
		"code": 40, 
		"message": "download failed", 
		"result":{"detections":[]}
	}, 
	{ 
		"code": 0, 
		"message": "success", 
		"result": { "detections":[ { "index": 0, "pts": [[ 1259, 70 ], [ 1281, 70 ], [ 1281, 99 ], [ 1259, 99 ]], "class": "", "score": 0 } ] } 
	}, 
	{ 
		"code": 0, 
		"message": "success", 
		"result": { "detections":[ { "index": 1, "class": "face", "pts": [ [ 225, 195 ], [ 351, 195 ], [ 351, 389 ], [ 225, 389 ] ], "score": 0.9971 } ] }
    }, 
	{ 
		"code": 0, 
		"message": "success", 
		"result": { "detections":[ { "index": 0, "class": "face", "pts": [ [ 1259, 70 ], [ 1281, 70 ], [ 1281, 99 ], [ 1259, 99 ] ], "score": 0.9455 } ] } 
	},
	{ 
		"code": 0, 
		"message": "success", 
		"result": { "detections":[ { "index": 1, "class": "face", "pts": [ [ 225, 195 ], [ 351, 195 ], [ 351, 389 ], [ 225, 389 ] ], "score": 0.9971 } ] }
    }, 
	{ 
		"code": 0, 
		"message": "success", 
		"result": { "detections":[ { "index": 0, "class": "face", "pts": [ [ 1259, 70 ], [ 1281, 70 ], [ 1281, 99 ], [ 1259, 99 ] ], "score": 0.9455 } ] } 
	}
	]
`
	xl := xlog.FromContextSafe(ctx)
	err = json.Unmarshal([]byte(result), &ret)
	if err != nil {
		xl.Errorf("json marshal error:%v", err.Error())
		return
	}
	xl.Debugf("test ret: %#v", ret)
	return
}
func (c *mockCl) PostFacexFeature1(ctx context.Context, args client.FacexFeatureReq, env client.EvalEnv) (ret []byte, err error) {
	return
}

/*func (c *mockCl) PostFacexFeature2(ctx context.Context, args client.FacexFeatureReq2) (ret client.FacexFeatureResp, err error) {
	return
}*/
func (c *mockCl) PostFacexCluster(ctx context.Context, args client.FacexClusterReq, env client.EvalEnv) (ret client.FacexClusterResp, err error) {
	return
}
