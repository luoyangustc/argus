package biz

import (
	"crypto/sha1"
	"encoding/hex"
	"encoding/json"

	"github.com/go-kit/kit/endpoint"

	"qiniu.com/argus/service/middleware"
	"qiniu.com/argus/service/service/biz"
	httptransport "qiniu.com/argus/service/transport"
)

type Evals struct{}

func NewEvals() *Evals { return &Evals{} }

func (evals *Evals) Make(method, host string, respSample interface{}) (endpoint.Endpoint, error) {
	// if evals.Redirect != nil {
	// 	if host1, ok := evals.Redirect[host]; ok {
	// 		host = host1
	// 	}
	// }
	return httptransport.MakeHttpClient(method, host, respSample)
}

////////////////////////////////////////////////////////////////////////////////

type ServiceEvalConfig struct {
	Id       string              `json:"id"` //标识原子服务,即相同id为同一原子服务,不同id为不同原子服务
	Name     string              `json:"name"`
	Version  string              `json:"version"`
	Host     string              `json:"host"`
	Redirect string              `json:"redirect"`
	Model    biz.EvalModelConfig `json:"model"`
}

type ServiceEvalDeployConfig struct {
	Modes []ServiceEvalDeployMode `json:"modes,omitempty"`
}

type ServiceEvalDeployMode struct {
	Device       string                       `json:"device,omitempty"`
	ProcessOnGPU [][]ServiceEvalDeployProcess `json:"process_on_gpu,omitempty"`
}

type ServiceEvalDeployProcess struct {
	Name string `json:"name"` // Eval Name
	Num  int    `json:"num,omitempty"`
}

//----------------------------------------------------------------------------//

type ServiceEvalInfo struct {
	Name    string
	Version string
}

type ServiceEvalSetter interface {
	Gen() (middleware.Service, error)
	Kernel() middleware.Service
	SetModel(biz.EvalModelConfig) ServiceEvalSetter
	GenId() ServiceEvalSetter
}

var _ ServiceEvalSetter = &serviceEvalSetter{}

type serviceEvalSetter struct {
	info ServiceEvalInfo
	*ServiceEvalConfig
	chain *middleware.ServiceChain
}

func NewServiceEvalSetter(
	info ServiceEvalInfo, conf *ServiceEvalConfig, chain *middleware.ServiceChain,
) ServiceEvalSetter {
	return &serviceEvalSetter{
		info:              info,
		ServiceEvalConfig: conf,
		chain:             chain,
	}
}

func (s serviceEvalSetter) Gen() (middleware.Service, error) {
	return s.chain.Gen()
}

func (s serviceEvalSetter) Kernel() middleware.Service {
	return s.chain.Kernel()
}

func (s *serviceEvalSetter) SetModel(model biz.EvalModelConfig) ServiceEvalSetter {
	s.ServiceEvalConfig.Model = model
	return s
}

//use func SetModel at first
func (s *serviceEvalSetter) GenId() ServiceEvalSetter {
	data, _ := json.Marshal(s.ServiceEvalConfig.Model)
	sha1 := sha1.New()
	sha1.Write(data)
	s.ServiceEvalConfig.Id = hex.EncodeToString(sha1.Sum(nil))
	return s
}
