package token

import (
	"fmt"
	"sync"
	"time"

	"golang.org/x/net/context"
)

type TokenProvider interface {
	GetToken() (ret string, err error)
}

type UserTokenProvider struct {
	token  *Token
	client UserClient
	lock   sync.Mutex
}

func NewUserTokenProvider(client UserClient) TokenProvider {
	return &UserTokenProvider{
		client: client,
	}
}

func (p *UserTokenProvider) GetToken() (ret string, err error) {
	t := p.token
	if t == nil || t.ExpiresIn(30*time.Second) {
		p.lock.Lock()
		defer p.lock.Unlock()

		if t == nil || t.ExpiresIn(30*time.Second) {
			var ut UserToken
			ut, err = p.client.CreateToken(context.Background())
			if err != nil {
				err = fmt.Errorf("Fail to refresh user token: %s", err.Error())
				return
			}

			p.token = &ut.Token
		}
	}

	ret = p.token.TokenID
	return
}

type ProjectTokenProvider struct {
	projectName string
	token       *Token
	client      ProjectClient
	lock        sync.Mutex
}

func NewProjectTokenProvider(projectName string, client ProjectClient) TokenProvider {
	return &ProjectTokenProvider{
		projectName: projectName,
		client:      client,
	}
}

func (p *ProjectTokenProvider) GetToken() (ret string, err error) {
	t := p.token
	if t == nil || t.ExpiresIn(30*time.Second) {
		p.lock.Lock()
		defer p.lock.Unlock()

		if t == nil || t.ExpiresIn(30*time.Second) {
			var pt ProjectToken
			pt, err = p.client.CreateToken(context.Background(), p.projectName)
			if err != nil {
				err = fmt.Errorf("Fail to refresh project token: %s", err.Error())
				return
			}

			p.token = &pt.Token
		}
	}

	ret = p.token.TokenID
	return
}
