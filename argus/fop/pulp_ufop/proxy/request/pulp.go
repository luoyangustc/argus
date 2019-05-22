package request

import (
	"net/url"

	"qiniu.com/argus/fop/pulp_ufop/proxy/client"
	"qiniu.com/argus/fop/pulp_ufop/proxy/cmd"
)

func pulp(client proxy_client.Client, urL, image string) ([]byte, error) {
	return client.PostForm(urL, url.Values{
		"image": []string{image},
	})
}

func init() {
	register(cmd.Pulp, handle(pulp))
}
