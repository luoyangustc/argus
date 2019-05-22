package censor

import (
	"encoding/json"

	"qiniu.com/argus/utility/evals"
	"qiniu.com/argus/utility/server"
)

func init() {
	var (
		es = ES{
			eP: "pulp", ePD: "pulp-detect",
			eTDP: "terror-predetect", eTD: "terror-detect", eTC: "terror-class", eTDPo: "terror-postdet",
			eFD: "facex-detect", eFF: "facex-feature-v2", eFF3: "facex-feature-v3", ePO: "politician", ePOu: "politician-u",
		}
	)
	server.RegisterEval(es.eP, func(cfg server.EvalConfig) interface{} { return evals.NewPulp(cfg.Host, cfg.Timeout) })
	server.RegisterEval(es.ePD, func(cfg server.EvalConfig) interface{} { return evals.NewPulpDetect(cfg.Host, cfg.Timeout) })
	server.RegisterEval(es.eTDP, func(cfg server.EvalConfig) interface{} { return evals.NewTerrorPreDetect(cfg.Host, cfg.Timeout) })
	server.RegisterEval(es.eTD, func(cfg server.EvalConfig) interface{} { return evals.NewTerrorDetect(cfg.Host, cfg.Timeout) })
	server.RegisterEval(es.eTC, func(cfg server.EvalConfig) interface{} { return evals.NewTerrorClassify(cfg.Host, cfg.Timeout) })
	server.RegisterEval(es.eTDPo, func(cfg server.EvalConfig) interface{} { return evals.NewTerrorPostDetect(cfg.Host, cfg.Timeout) })
	server.RegisterEval(es.eFD, func(cfg server.EvalConfig) interface{} { return evals.NewFaceDetect(cfg.Host, cfg.Timeout) })
	server.RegisterEval(es.eFF, func(cfg server.EvalConfig) interface{} { return evals.NewFaceFeature(cfg.Host, cfg.Timeout, "-v2") })
	server.RegisterEval(es.eFF3, func(cfg server.EvalConfig) interface{} { return evals.NewFaceFeature(cfg.Host, cfg.Timeout, "-v3") })
	server.RegisterEval(es.ePO, func(cfg server.EvalConfig) interface{} { return evals.NewPolitician(cfg.Host, cfg.Timeout, "") })
	server.RegisterEval(es.ePOu, func(cfg server.EvalConfig) interface{} { return evals.NewPolitician(cfg.Host, cfg.Timeout, "-u") })

	server.RegisterHandler("/image/censor", &Service{ES: es})
}

type ES struct {
	eP, ePD                   string
	eTDP, eTD, eTC, eTDPo     string
	eFD, eFF, eFF3, ePO, ePOu string
}

type Config struct {
	TerrorThreshold          float32   `json:"terror_threshold"`
	PulpReviewThreshold      float32   `json:"pulp_review_threshold"`
	PoliticianThreshold      []float32 `json:"politician_threshold"`
	PoliticianFeatureVersion string    `json:"politician_feature_version"`
	PoliticianUpdate         bool      `json:"politician_update"`
}

type Service struct {
	ES
	Config
	server.IServer

	ePulp               evals.IPulp
	ePulpDetect         evals.IPulpDetect
	eTerrorPreDet       evals.ITerrorPreDetect
	eTerrorDet          evals.ITerrorDetect
	eTerrorClassify     evals.ITerrorClassify
	eTerrorPostDet      evals.ITerrorPostDetect
	eFaceDet            evals.IFaceDetect
	eFaceFeature        evals.IFaceFeature
	ePoliticianFFeature evals.IFaceFeature
	ePolitician         evals.IPolitician
}

func (s *Service) Init(msg json.RawMessage, is server.IServer) interface{} {
	var cfg Config
	_ = json.Unmarshal(msg, &cfg)
	s.Config = cfg
	s.IServer = is

	s.ePulp = is.GetEval(s.ES.eP).(evals.IPulp)
	s.ePulpDetect = is.GetEval(s.ES.ePD).(evals.IPulpDetect)
	s.eTerrorPreDet = is.GetEval(s.ES.eTDP).(evals.ITerrorPreDetect)
	s.eTerrorDet = is.GetEval(s.ES.eTD).(evals.ITerrorDetect)
	s.eTerrorClassify = is.GetEval(s.ES.eTC).(evals.ITerrorClassify)
	s.eTerrorPostDet = is.GetEval(s.ES.eTDPo).(evals.ITerrorPostDetect)
	s.eFaceDet = is.GetEval(s.ES.eFD).(evals.IFaceDetect)
	s.eFaceFeature = is.GetEval(s.ES.eFF).(evals.IFaceFeature)

	s.ePolitician = is.GetEval(s.ES.ePO).(evals.IPolitician)
	if s.Config.PoliticianUpdate {
		s.ePolitician = is.GetEval(s.ES.ePOu).(evals.IPolitician)
	}

	s.ePoliticianFFeature = is.GetEval(s.ES.eFF).(evals.IFaceFeature)
	if s.Config.PoliticianFeatureVersion == "v3" {
		s.ePoliticianFFeature = is.GetEval(s.ES.eFF3).(evals.IFaceFeature)
	}
	return s
}
