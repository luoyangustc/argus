package sdk

import (
	"net/http"

	"github.com/qbox/ke-base/sdk/account"
	"github.com/qbox/ke-base/sdk/cert"
	"github.com/qbox/ke-base/sdk/configmap"
	"github.com/qbox/ke-base/sdk/domain"
	"github.com/qbox/ke-base/sdk/lb"
	"github.com/qbox/ke-base/sdk/microservice"
	"github.com/qbox/ke-base/sdk/token"
	"github.com/qbox/ke-base/sdk/util"
	"github.com/qbox/ke-base/sdk/volume"
)

type Config struct {
	Username  string
	Password  string
	Host      string
	Transport http.RoundTripper
}

type ClientSet struct {
	config    Config
	transport http.RoundTripper
}

func New(config Config) *ClientSet {
	trans := config.Transport
	if trans == nil {
		trans = http.DefaultTransport
	}

	return &ClientSet{
		config: config,
		transport: token.NewAutoRefreshTransport(
			config.Host,
			config.Username,
			config.Password,
			trans),
	}
}

func (p *ClientSet) Account() account.Client {
	return account.NewWithTransport(p.config.Host, p.transport)
}

func (p *ClientSet) Cert() cert.Client {
	return cert.NewWithTransport(p.config.Host, p.transport)
}

func (p *ClientSet) Domain(region string) domain.Client {
	return domain.NewWithTransport(p.regionHost(region), p.transport)
}

func (p *ClientSet) LoadBalance(region string) lb.Client {
	return lb.NewWithTransport(p.regionHost(region), p.transport)
}

func (p *ClientSet) ConfigMap(region string) configmap.Client {
	return configmap.NewWithTransport(p.regionHost(region), p.transport)
}

func (p *ClientSet) Volume(region string) volume.Client {
	return volume.NewWithTransport(p.regionHost(region), p.transport)
}

func (p *ClientSet) MicroService(region string) microservice.Client {
	return microservice.NewWithTransport(p.regionHost(region), p.transport)
}

func (p *ClientSet) regionHost(region string) string {
	return util.CleanHost(p.config.Host) + "/regions/" + region
}
