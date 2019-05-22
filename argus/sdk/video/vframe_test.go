package video

import (
	"bytes"
	"context"
	"encoding/json"
	"io/ioutil"
	"net/http"
	"testing"
	"time"

	rpc "github.com/qiniu/rpc.v3"
	"github.com/stretchr/testify/assert"
)

func TestHttpCallback(t *testing.T) {
	type _result struct {
		Code    int    `json:"code"`
		Message string `json:"message"`
		Result  string `json:"result"`
	}
	var (
		ctx   = context.Background()
		vr    = &vframeV2{}
		image = []byte("this is test image")
	)

	process := func(ctx context.Context, body []byte) (interface{}, error) {
		assert.Equal(t, string(image), string(body))
		return _result{
			Code:    0,
			Message: "OK",
			Result:  "test result",
		}, nil
	}
	uri, err := vr.genHttpRpc(ctx, process)
	assert.Nil(t, err)
	time.Sleep(100 * time.Millisecond)
	client := rpc.Client{
		Client: &http.Client{},
	}
	resp, err := client.DoRequestWith(ctx, "POST", uri, "octet-stream", bytes.NewReader(image), len(image))
	assert.Nil(t, err)
	defer resp.Body.Close()
	assert.Equal(t, resp.StatusCode, 200)
	buf, err := ioutil.ReadAll(resp.Body)
	assert.Nil(t, err)
	var result _result
	assert.Nil(t, json.Unmarshal(buf, &result))
	assert.Equal(t, 0, result.Code)
	assert.Equal(t, "OK", result.Message)
	assert.Equal(t, "test result", result.Result)
}
