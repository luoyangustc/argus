package token

import (
	"time"

	"github.com/qbox/ke-base/sdk/proto"
)

// UserToken return for get user's token
type UserToken struct {
	Token         Token          `json:"token"`
	User          proto.UserInfo `json:"user"`
	SSOLoginToken string         `json:"ssoLoginToken,omitempty"`
}

type ProjectToken struct {
	Token   Token          `json:"token"`
	Project string         `json:"project"`
	Roles   []string       `json:"roles"`
	User    proto.UserInfo `json:"user"`
}

type Token struct {
	TokenID   string    `json:"id"`
	ExpiresAt time.Time `json:"expires_at"`
	IssuedAt  time.Time `json:"issued_at"`
}

func (p *Token) Expired() bool {
	return p.ExpiresIn(0)
}

func (p *Token) ExpiresIn(d time.Duration) bool {
	return time.Now().Add(d).After(p.ExpiresAt)
}
