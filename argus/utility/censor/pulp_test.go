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

type mockPulp struct{}

func (e mockPulp) Eval(
	ctx context.Context, req evals.PulpReq, uid, utype uint32,
) (resp evals.PulpResp, err error) {
	ret := `
			{
    			"code": 0,
    			"message": "",
    			"result": {
        			"confidences": [
            			{
                			"class": "norm",
                			"index": 2,
                			"score": 0.6
            			},
						{
                			"class": "sexy",
                			"index": 1,
                			"score": 0.5
            			},
						{
                			"class": "pulp",
                			"index": 0,
                			"score": 0.4
            			}
        			]
    			}
			}
	`
	err = json.Unmarshal([]byte(ret), &resp)
	return
}

var _ server.IImageParse = mockImageParse{}

type mockImageParse struct{}

func (e mockImageParse) ParseImage(ctx context.Context, uri string) (img server.Image, err error) {

	img.Format = "png"
	img.Width = 176
	img.Height = 201

	if uri == "http://test.image1.jpg" {
		img.Format = "png"
		img.Width = 300
		img.Height = 201
	}
	return
}

func TestPulp(t *testing.T) {
	srv := &Service{
		ES: ES{
			eP: "p",
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
		ePulp: mockPulp{},
	}
	ctx := server.NewHTContext(t, srv)

	req := ctx.Request("POST", "http://test.com/v1/pulp")
	req.WithBody("application/json", "{\"data\": {\"uri\": \"http://test.image.jpg\"}}")
	req.WithHeader("Authorization", []string{"QiniuStub uid=1&ut=4"}...)
	resp := req.Ret(200)

	var pResp PulpResp
	err := json.Unmarshal(resp.RawBody, &pResp)

	assert.NoError(t, err)
	assert.Equal(t, 0, pResp.Code)
	assert.Equal(t, 2, pResp.Result.Label)
	assert.True(t, math.Abs(float64(pResp.Result.Score)-0.40449440) <= 0.00000005)
}
