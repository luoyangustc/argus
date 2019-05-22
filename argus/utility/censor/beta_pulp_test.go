package censor

import (
	"context"
	"encoding/json"
	"math"
	"testing"

	"github.com/stretchr/testify/assert"
	"qiniu.com/argus/utility/evals"
	"qiniu.com/argus/utility/server"
)

type mockPulpDetect struct{}

func (e mockPulpDetect) Eval(
	ctx context.Context, req evals.PulpDetectReq, uid, utype uint32,
) (resp evals.PulpDetectResp, err error) {
	ret := `
			{
    			"code": 0,
    			"message": "",
    			"result": {
        			"detections": [
            			{
                			"class": "tits",
                			"index": 2,
							"score": 0.65,
							"pts":[[112,334],[222,334],[222,456],[112,456]]
            			}
        			]
    			}
			}
	`
	err = json.Unmarshal([]byte(ret), &resp)
	return
}

var _ server.IImageParse = mockImageParse{}

func TestBetaPulp(t *testing.T) {
	srv := &Service{
		ES: ES{
			eP: "p", ePD: "ps",
		},
		Config: Config{
			TerrorThreshold:     0.25,
			PoliticianThreshold: []float32{0.6, 0.66, 0.72},
			PulpReviewThreshold: 0.89,
		},
		IServer: server.MockStaticServer{
			ParseImageF: func(ctx context.Context, uri string) (server.Image, error) {
				return mockImageParse{}.ParseImage(ctx, uri)
			},
		},
		ePulp:       mockPulp{},
		ePulpDetect: mockPulpDetect{},
	}
	ctx := server.NewHTContext(t, srv)

	req := ctx.Request("POST", "http://test.com/v1/beta/pulp")
	req.WithBody("application/json", "{\"data\": {\"uri\": \"http://test.image.jpg\"}}")
	req.WithHeader("Authorization", []string{"QiniuStub uid=1&ut=4"}...)
	resp := req.Ret(200)

	var pResp PulpResp
	err := json.Unmarshal(resp.RawBody, &pResp)

	assert.NoError(t, err)
	assert.Equal(t, 0, pResp.Code)
	assert.Equal(t, 0, pResp.Result.Label)
	print(pResp.Result.Score)
	assert.True(t, math.Abs(float64(pResp.Result.Score)-0.0750000) <= 0.000005)
}
