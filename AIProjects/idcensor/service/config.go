package idcensor

import "strings"

type AccessKey struct {
	ID     string
	Secret []byte
}

func NewAccessKey(id, secret string) *AccessKey {
	return &AccessKey{
		ID:     id,
		Secret: []byte(secret),
	}
}

type Auth struct {
	Scope       string
	AccessKeyID string
	Signature   string
}

func NewAuth(header string) *Auth {
	auth := &Auth{}

	prefix := "Qiniu "
	if strings.HasPrefix(header, prefix) {
		auth.Scope = "Qiniu"
		header = strings.TrimPrefix(header, prefix)
	}

	tmp := strings.SplitN(header, ":", 2)
	if len(tmp) == 2 {
		auth.AccessKeyID = strings.TrimSpace(tmp[0])
		auth.Signature = strings.TrimSpace(tmp[1])
	}

	return auth
}

func (auth *Auth) IsValid() bool {
	return auth.Scope == "Qiniu" && auth.AccessKeyID != "" && auth.Signature != ""
}
