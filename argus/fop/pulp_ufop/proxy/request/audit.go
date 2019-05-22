package request

import (
	"encoding/json"
	"errors"

	"qiniu.com/argus/fop/pulp_ufop/proxy/client"
	"qiniu.com/argus/fop/pulp_ufop/proxy/cmd"
)

func audit(client proxy_client.Client, url, image string) ([]byte, error) {
	ch := make(chan [2]string, 1)

	go func() {
		d, r := pulp(client, url, image)
		if r != nil {
			ch <- [2]string{"", r.Error()}
			return
		}
		ch <- [2]string{string(d), ""}
	}()

	var err error
	o := map[string]string{}

	d, err := client.Get(image+"?terror", nil)

	for {
		r := <-ch
		if err != nil {
			break
		}

		if e := r[1]; e != "" {
			err = errors.New(e)
			break
		}

		o[cmd.Audit] = string(d)
		o[cmd.Pulp] = string(r[0])

		break
	}

	close(ch)

	if err != nil {
		return nil, err
	}
	return json.Marshal(o)
}

func init() {
	register(cmd.Audit, handle(audit))
}
