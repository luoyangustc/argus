package token

import (
	"sync"
)

type tokenStore struct {
	tokenMap          map[string]TokenProvider
	userTokenProvider TokenProvider
	projectClient     ProjectClient
	lock              sync.Mutex
}

func (p *tokenStore) GetToken(projectName string) (token string, err error) {
	tp := p.GetTokenProvider(projectName)

	return tp.GetToken()
}

func (p *tokenStore) GetTokenProvider(projectName string) (ret TokenProvider) {
	if projectName == "" {
		return p.userTokenProvider
	}

	p.lock.Lock()
	defer p.lock.Unlock()

	ret, ok := p.tokenMap[projectName]
	if !ok {
		ret = NewProjectTokenProvider(projectName, p.projectClient)
		p.tokenMap[projectName] = ret
	}

	return
}
