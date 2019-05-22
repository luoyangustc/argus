package request

import (
	"errors"

	"qiniu.com/argus/fop/pulp_ufop/proxy/client"
)

type handle func(proxy_client.Client, string, string) ([]byte, error)

var handlers = map[string]handle{}

func register(cmd string, h handle) {
	handlers[cmd] = h
}

func Do(cmd, url, image string, client proxy_client.Client) ([]byte, error) {
	h, ok := handlers[cmd]

	if !ok {
		return nil, errors.New("not found cmd")
	}

	return h(client, url, image)
}
