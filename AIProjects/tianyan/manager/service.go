package manager

import (
	"context"

	xlog "github.com/qiniu/xlog.v1"
	"qiniu.com/argus/AIProjects/tianyan/serving"
)

type Config struct {
	serving.EvalConfig `json:"default"`
	Evals              map[string]serving.EvalConfig `json:"evals"`
	FaceLimit          int                           `json:"face_limit"`
	SearchThreshold    float32                       `json:"search_threshold"`
	ClusterThreshold   float32                       `json:"cluster_threshold"`
	MinFaceWidth       int                           `json:"min_face_width"`
	MinFaceHeight      int                           `json:"min_face_height"`
}

func (c Config) Get(cmd string) serving.EvalConfig {
	if cc, ok := c.Evals[cmd]; ok {
		if cc.Host == "" {
			cc.Host = c.EvalConfig.Host
		}
		if cc.Timeout == 0 {
			cc.Timeout = c.EvalConfig.Timeout
		}
		cc.Timeout = c.Timeout
		return cc
	}
	return c.EvalConfig
}

type Service struct {
	Config
	Manager

	serving.FaceDetect
	serving.FaceFeatureV2
	serving.FeatureSearch
}

func New(c Config, mgr Manager) (*Service, error) {
	srv := &Service{
		Config:  c,
		Manager: mgr,
	}
	if srv.SearchThreshold == 0 {
		srv.SearchThreshold = defaultSearchThreshold
	}
	if srv.ClusterThreshold == 0 {
		srv.ClusterThreshold = defaultClusterThreshold
	}

	if srv.FaceLimit == 0 {
		srv.FaceLimit = defaultFaceLimit
	}

	if srv.MinFaceWidth == 0 {
		srv.MinFaceWidth = defaultMinFaceWidth
	}

	if srv.MinFaceHeight == 0 {
		srv.MinFaceHeight = defaultMinFaceHeight
	}

	const (
		_CmdFaceDetect    = "facex-detect"
		_CmdFaceFeature   = "facex-feature"
		_CmdFeatureSearch = "feature-search"
	)
	{
		conf := c.Get(_CmdFaceDetect)
		srv.FaceDetect = serving.NewFaceDetect(conf)
	}
	{
		conf := c.Get(_CmdFaceFeature)
		srv.FaceFeatureV2 = serving.NewFaceFeatureV2(conf)
	}
	{
		conf := c.Get(_CmdFeatureSearch)
		srv.FeatureSearch = serving.NewFeatureSearch(conf)
	}

	xl := xlog.NewDummy()
	ctx := xlog.NewContext(context.Background(), xl)
	if err := srv.initGroups(ctx); err != nil {
		return nil, err
	}
	return srv, nil
}
