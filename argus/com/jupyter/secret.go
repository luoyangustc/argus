package jupyter

import (
	"crypto/sha1"
	"encoding/hex"

	"qiniu.com/argus/com/util"
)

const saltLength = 12
const secretAlgorithm = "sha1"

// Secret password for jupyter notebook/lab
type Secret struct {
	Algorithm string
	Salt      string
	Hash      string
	phrase    string
}

func NewSecret(phrase string) *Secret {
	return &Secret{
		Algorithm: secretAlgorithm,
		phrase:    phrase,
	}
}

// Encode password phrase into hash
func (s *Secret) Encode() {
	if s.Salt == "" {
		s.Salt, _ = util.RandomString(saltLength, 16)
	}
	if s.Algorithm == "" {
		s.Algorithm = secretAlgorithm
	}
	hash := sha1.New()
	hash.Write([]byte(s.phrase))
	hash.Write([]byte(s.Salt))
	s.Hash = hex.EncodeToString(hash.Sum(nil))
}

func (s *Secret) String() string {
	if s.phrase != "" && s.Hash == "" {
		s.Encode()
	}
	return s.Algorithm + ":" + s.Salt + ":" + s.Hash
}
