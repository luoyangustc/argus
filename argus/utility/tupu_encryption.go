package utility

import (
	"crypto"
	"crypto/rand"
	"crypto/rsa"
	"crypto/sha256"
	"crypto/x509"
	"encoding/pem"
	"errors"
)

func GenSha256(strOrigin string) []byte {
	hash := sha256.New()
	hash.Write([]byte(strOrigin))
	return hash.Sum(nil)
}

// Rsa加密 (内存模式)
func RsaEncryptByMemory(origData []byte, publicKey []byte) ([]byte, error) {
	block, _ := pem.Decode(publicKey)
	if block == nil {
		return nil, errors.New("public key error")
	}

	pubInterface, err := x509.ParsePKIXPublicKey(block.Bytes)
	if err != nil {
		return nil, err
	}

	pub := pubInterface.(*rsa.PublicKey)
	return rsa.EncryptPKCS1v15(rand.Reader, pub, origData)
}

// Rsa解密 (内存模式)
func RsaPkcs1DecryptByMemory(ciphertext []byte, privateKey []byte) ([]byte, error) {
	block, _ := pem.Decode(privateKey)
	if block == nil {
		return nil, errors.New("private key error!")
	}

	priv, err := x509.ParsePKCS1PrivateKey(block.Bytes)
	if err != nil {
		return nil, err
	}

	return rsa.DecryptPKCS1v15(rand.Reader, priv, ciphertext)
}

//rsa签字 (内存模式)
func RsaPkcs8SignByMemory(Digest []byte, hash crypto.Hash, privateKey []byte) ([]byte, error) {
	block, _ := pem.Decode(privateKey)
	if block == nil {
		return nil, errors.New("Decode private key error!")
	}

	priv, err := x509.ParsePKCS8PrivateKey(block.Bytes)
	if err != nil {
		return nil, errors.New("Parse PKCS8 private key is error")
	}

	privK := priv.(*rsa.PrivateKey)
	return rsa.SignPKCS1v15(rand.Reader, privK, hash, Digest)
}

//rsa验证 (内存模式)
func ReaVerifyByMemory(Digest, Sign []byte, hash crypto.Hash, publicKey []byte) error {
	block, _ := pem.Decode(publicKey)
	if block == nil {
		return errors.New("public key error")
	}

	pubInterface, err := x509.ParsePKIXPublicKey(block.Bytes)
	if err != nil {
		return err
	}

	pub := pubInterface.(*rsa.PublicKey)

	return rsa.VerifyPKCS1v15(pub, hash, Digest, Sign)
}
