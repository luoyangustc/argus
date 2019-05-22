package token

import (
	"net/http"
	"strings"
)

func NewAutoRefreshTransport(authHost string, username, password string, trans http.RoundTripper) http.RoundTripper {
	if trans == nil {
		trans = http.DefaultTransport
	}

	userClient := NewUserTokenClient(authHost, username, password)
	utProvider := NewUserTokenProvider(userClient)

	utpTransport := NewTokenProviderTransport(utProvider, nil)
	projectClient := NewProjectTokenClient(authHost, utpTransport)

	return &AutoRefreshTransport{
		tokenStore: &tokenStore{
			tokenMap:          make(map[string]TokenProvider),
			userTokenProvider: utProvider,
			projectClient:     projectClient,
		},
		transport: trans,
	}
}

type AutoRefreshTransport struct {
	*tokenStore
	transport http.RoundTripper
}

func (p *AutoRefreshTransport) RoundTrip(req *http.Request) (resp *http.Response, err error) {
	project := extractProject(req.URL.Path)

	token, err := p.tokenStore.GetToken(project)
	if err != nil {
		return
	}

	req.Header.Set("X-Auth-Token", token)
	resp, err = p.transport.RoundTrip(req)
	return
}

func extractProject(path string) (project string) {
	path = strings.TrimSpace(path)
	path = strings.TrimPrefix(path, "/")
	path = strings.TrimSuffix(path, "/")

	components := strings.Split(path, "/")
	stat := 0
	for i := 0; i < len(components); i++ {
		current := components[i]
		switch stat {
		case 0:
			if strings.HasPrefix(current, "v") {
				stat++
			}
			continue
		case 1:
			if current == "projects" {
				stat++
			} else {
				stat = 0
			}
			continue
		case 2:
			project = current
			return
		default:
			return
		}
	}

	return
}

func NewTokenProviderTransport(provider TokenProvider, trans http.RoundTripper) http.RoundTripper {
	if trans == nil {
		trans = http.DefaultTransport
	}

	return &TokenProviderTransport{
		provider:  provider,
		transport: trans,
	}
}

type TokenProviderTransport struct {
	provider  TokenProvider
	transport http.RoundTripper
}

func (p *TokenProviderTransport) RoundTrip(req *http.Request) (resp *http.Response, err error) {
	token, err := p.provider.GetToken()
	if err != nil {
		return
	}

	req.Header.Set("X-Auth-Token", token)
	resp, err = p.transport.RoundTrip(req)
	return
}
