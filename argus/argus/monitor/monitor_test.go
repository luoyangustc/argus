package monitor

import (
	"errors"
	"github.com/stretchr/testify/assert"
	"io/ioutil"
	"net/http"
	"net/http/httptest"
	"qiniupkg.com/http/httputil.v2"
	"testing"
	"time"
)

func TestMonitor_Handler(t *testing.T) {
	assert := assert.New(t)
	s := httptest.NewServer(Handler())
	defer s.Close()
	ResponseTime("face", nil, time.Second)
	ResponseTime("object", nil, time.Second*2)
	ResponseTime("face", httputil.NewError(http.StatusBadRequest, "number of images exceeded 10"), time.Second*35)
	ResponseTime("face", errors.New("xx"), time.Second*35)
	InferenceResponseTime("PostScene", nil, time.Second)
	body, err := httpGet(s.URL)
	assert.Nil(err)
	assert.Contains(body, `ava_argus_gate_response_time_bucket{api="face",code="400",le="60000"} 1`)
	assert.Contains(body, `ava_argus_gate_response_time_bucket{api="face",code="599",le="60000"} 1`)
	assert.Contains(body, `ava_argus_gate_inference_response_time_bucket{api="PostScene",code="200",le="1000"} 1`)
	// fmt.Println(body)
}

func httpGet(url string) (body string, err error) {
	resp, err := http.Get(url)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()
	buf, err := ioutil.ReadAll(resp.Body)
	if err != nil {
		return "", err
	}
	return string(buf), nil
}
