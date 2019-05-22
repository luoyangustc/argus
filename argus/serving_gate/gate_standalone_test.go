package gate

import (
	"context"
	"testing"

	"qbox.us/net/httputil"

	"github.com/stretchr/testify.v2/assert"
	"qiniu.com/argus/atserving/model"
)

func TestStandaloneGate(t *testing.T) {

	evals := NewEvals()
	evals.SetAppMetadataDefault(model.ConfigAppMetadata{
		Public: true,
	})
	c := NewEvalClients()
	c.SetHosts(map[string]map[string][]string{
		"face": map[string][]string{
			"v1": []string{
				"http://hostst.ssd",
				"http://hostst1.ssd",
			},
		},
	}, 0)
	g := NewGateStandalone(evals, c)
	_, _, _, _, err := g.Eval(
		context.Background(), model.EvalRequest{
			Cmd: "feature",
		},
	)
	assert.Equal(t, "no valid hosts", err.Error())

	_, _, _, _, err = g.Eval(
		context.Background(), model.EvalRequest{
			Cmd:     "face",
			Version: sp("v1"),
			Data: model.Resource{
				URI: model.STRING("http://image.jpg"),
			},
		},
	)

	code, _ := httputil.DetectError(err)
	assert.Equal(t, 599, code)
}

func TestStandaloneEvals(t *testing.T) {

	c := NewEvalClients()
	c.SetHosts(map[string]map[string][]string{
		"face": map[string][]string{
			"v1": []string{
				"http://hostst.ssd",
				"http://hostst1.ssd",
			},
		},
	}, 0)
	cli := c.GetClient("face", sp("v1"))
	assert.NotNil(t, cli)

	cli = c.GetClient("feature", nil)
	assert.Nil(t, cli)
}
