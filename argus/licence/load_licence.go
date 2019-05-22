package licence

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io/ioutil"
	"os"

	"golang.org/x/crypto/openpgp"
	"golang.org/x/crypto/openpgp/clearsign"
)

// LoadLicence ...
//
func LoadLicence(fileName string) (Licence, error) {
	keyring, err := openpgp.ReadArmoredKeyRing(bytes.NewBufferString(pubkey))
	if err != nil {
		return nil, err
	}
	f, err := os.Open(fileName)
	if err != nil {
		return nil, err
	}

	alldata, err := ioutil.ReadAll(f)
	if err != nil {
		return nil, err
	}

	b, _ := clearsign.Decode(alldata)
	if b == nil {
		return nil, fmt.Errorf("clearsign.Decode error")
	}
	if _, err := openpgp.CheckDetachedSignature(keyring, bytes.NewBuffer(b.Bytes), b.ArmoredSignature.Body); err != nil {
		return nil, err
	}

	lic := Licence{}
	err = json.Unmarshal(b.Bytes, &lic)
	if err != nil {
		return nil, err
	}
	return lic, nil
}
