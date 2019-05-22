package proxy_config

import (
	"io/ioutil"
	"os"
	"reflect"
	"testing"
)

func assert(expect, actual interface{}, t *testing.T) {
	switch actual.(type) {
	default:
		t.Fatal("unrecognized type:", reflect.TypeOf(expect))
	case string:
		if expect.(string) != actual.(string) {
			goto ERROR
		}
		return
	case int:
		if expect.(int) != actual.(int) {
			goto ERROR
		}
		return
	case uint32:
		if expect.(uint32) != actual.(uint32) {
			goto ERROR
		}
		return
	}
ERROR:
	{
		t.Error("expect ", expect, ", actual ", actual)
		return
	}

}

func TestConfig(t *testing.T) {
	data := `
	{
		"port": 9090,
		"max_concurrent":1000,
		"bucket": {
			"ak":"ak",
			"sk":"sk",
			"name":"name",
			"domain":"domain"
		},
		"auth": {
			"ak":"aak",
			"sk":"ask"
		},
		"cmds": [
		{
			"name":"pulp",
			"url":"pulp.url"
		},
		{
			"name":"facex",
			"url":"facex.url"
		}
		]
	}`
	tmpfile, err := ioutil.TempFile("", "ufop.proxy.test")
	if err != nil {
		t.Fatal("cannot create tmp file", err)
	}

	defer os.Remove(tmpfile.Name())

	if _, err := tmpfile.Write([]byte(data)); err != nil {
		t.Fatal("write data to tmpfile", err)
	}

	config, err := LoadFromFile(tmpfile.Name())

	bucket, cmds, auth := &config.Bucket, config.Cmds, config.Auth

	assert("ak", bucket.Ak, t)
	assert("sk", bucket.Sk, t)
	assert("aak", auth.Ak, t)
	assert("ask", auth.Sk, t)
	assert("name", bucket.Name, t)
	assert("domain", bucket.Domain, t)
	assert(len(cmds), 2, t)
	assert("pulp", cmds[0].Name, t)
	assert("pulp.url", cmds[0].Url, t)
	assert("facex", cmds[1].Name, t)
	assert("facex.url", cmds[1].Url, t)
	assert(9090, config.Port, t)
	assert(uint32(1000), config.MaxConcurrent, t)

	if err := tmpfile.Close(); err != nil {
		t.Fatal("close tmpfile ", err)
	}
}
