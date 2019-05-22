package ops

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/qiniu/rpc.v3"
	"github.com/stretchr/testify/assert"
	"qiniu.com/argus/utility/censor"
	"qiniu.com/argus/video"
)

func TestOpTerrorComplex(t *testing.T) {
	params := video.OPParams{
		Other: struct {
			Detail bool `json:"detail"`
		}{
			Detail: true,
		},
	}

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		resp := censor.TerrorComplexResp{}
		resp.Result.Label = 1
		resp.Result.Classes = append(resp.Result.Classes, struct {
			Class string  `json:"class,omitempty"`
			Score float32 `json:"score,omitempty"`
		}{
			Class: "bomb",
			Score: 0.9997,
		})
		resp.Result.Classes = append(resp.Result.Classes, struct {
			Class string  `json:"class,omitempty"`
			Score float32 `json:"score,omitempty"`
		}{
			Class: "guns",
			Score: 0.9797,
		})
		resp.Result.Score = 0.9997
		resp.Result.Review = false

		js, _ := json.Marshal(resp)
		w.Write(js)
	}))

	eval := NewEvalTerrorComplex(params)
	resp, err := eval(context.Background(), &rpc.DefaultClient, srv.URL, "")
	assert.Nil(t, err)
	assert.NotNil(t, resp)
	assert.Equal(t, resp.(TerrorComplexResult).Len(), 1)
	label, score, flag := resp.(TerrorComplexResult).Parse(0)
	assert.Equal(t, label, "1")
	assert.Equal(t, score, float32(0.9997))
	assert.Equal(t, flag, true)
}
