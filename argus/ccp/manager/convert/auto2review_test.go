package convert

import (
	"context"
	"encoding/json"
	"fmt"
	"sort"
	"testing"

	"github.com/stretchr/testify/assert"
	"qiniu.com/argus/ccp/manager/client"
	"qiniu.com/argus/ccp/manager/proto"
	"qiniu.com/argus/censor/biz"
	"qiniu.com/argus/utility/censor"
)

func TestAuto2Review(t *testing.T) {

	et := ConvPfopOld2Review(&proto.Rule{}, "qiniu://abc.png", &PfopImageResultOld{
		Disable: true,
		Result: censor.ImageCensorResp{
			Code: 0,
			Result: struct {
				Label   int                            `json:"label"`
				Score   float32                        `json:"score"`
				Review  bool                           `json:"review"`
				Details []censor.ImageCensorDetailResp `json:"details"`
			}{
				Label:  0,
				Score:  0.9,
				Review: false,
			},
		},
	})
	assert.NotNil(t, et)
	assert.Equal(t, "IMAGE", et.MimeType)
	assert.Equal(t, "qiniu://abc.png", et.URIGet)
	assert.Equal(t, "AUTOMATIC", et.Original.Source)
	assert.Equal(t, SUG_DISABLED, et.Original.Suggestion)

	et0 := ConvPfopOld2Review(&proto.Rule{}, "qiniu://abc.png", &PfopImageResultOld{
		Disable: false,
		Result: censor.ImageCensorResp{
			Code: 0,
			Result: struct {
				Label   int                            `json:"label"`
				Score   float32                        `json:"score"`
				Review  bool                           `json:"review"`
				Details []censor.ImageCensorDetailResp `json:"details"`
			}{
				Label:  0,
				Score:  0.9,
				Review: false,
			},
		},
	})
	assert.NotNil(t, et0)
	assert.Equal(t, "IMAGE", et0.MimeType)
	assert.Equal(t, "qiniu://abc.png", et0.URIGet)
	assert.Equal(t, "AUTOMATIC", et0.Original.Source)
	assert.Equal(t, SUG_PASS, et0.Original.Suggestion)

	et2 := ConvPfop2Review(&proto.Rule{}, "qiniu://abc.png", &PfopImageResult{
		Disable: false,
		Result: biz.CensorResponse{
			Code:       0,
			Suggestion: biz.BLOCK,
		},
	})
	assert.NotNil(t, et2)
	assert.Equal(t, "IMAGE", et2.MimeType)
	assert.Equal(t, "qiniu://abc.png", et2.URIGet)
	assert.Equal(t, "AUTOMATIC", et2.Original.Source)
	assert.Equal(t, SUG_BLOCK, et2.Original.Suggestion)

	et3 := ConvPfop2Review(&proto.Rule{}, "qiniu://abc.png", &PfopImageResult{
		Disable: true,
		Result: biz.CensorResponse{
			Code:       0,
			Suggestion: biz.BLOCK,
		},
	})
	assert.NotNil(t, et3)
	assert.Equal(t, "IMAGE", et3.MimeType)
	assert.Equal(t, "qiniu://abc.png", et3.URIGet)
	assert.Equal(t, "AUTOMATIC", et3.Original.Source)
	assert.Equal(t, SUG_DISABLED, et3.Original.Suggestion)

	result := struct {
		UID    uint32 `json:"uid"`
		Bucket string `json:"bucket"`
		Key    string `json:"key"`
		Result struct {
			Result PfopVideoResult `json:"result"`
		} `json:"result"`
	}{}
	_ = json.Unmarshal([]byte(
		`{"uid":1380538984,"bucket":"argus-bcp-test","key":"2movie.mp4","result":{"result":{"disable":false,"result":{"scenes":{"pulp":{"segments":[{"cuts":[{"offset":0,"result":{"confidences":[{"class":"normal","index":2,"score":0.99856997},{"class":"sexy","index":1,"score":0.00030006122},{"class":"pulp","index":0,"score":0.00007848756}],"label":2,"review":false,"score":0.99856997},"suggestion":"pass"},{"offset":10143,"result":{"confidences":[{"class":"normal","index":2,"score":0.9988818},{"class":"sexy","index":1,"score":0.0002424714},{"class":"pulp","index":0,"score":0.000053464726}],"label":2,"review":false,"score":0.9988818},"suggestion":"pass"}],"offset_begin":0,"offset_end":10143,"suggestion":"pass"}],"suggestion":"pass"}},"suggestion":"pass"}}}}`,
	), &result)
	rstr, _ := json.Marshal(result)
	fmt.Println(string(rstr))

	ConvPfopVideo2Review(&proto.Rule{}, "qiniu:///argus-bcp-test/2movie.mp4", &result.Result.Result)

	tstr := `{"code":200,"mimetype":"image","result":{"code":200,"message":"OK","suggestion":"pass","scenes":{"politician":{"suggestion":"pass","result":{"label":"normal","faces":null}},"pulp":{"suggestion":"pass","result":{"label":"normal","score":0.99948424}},"terror":{"suggestion":"pass","result":{"label":"normal","score":0.960861}}}}}`
	rrr := AutoResult{}
	_ = json.Unmarshal([]byte(tstr), &rrr)
	eee := convBjob2ReviewItem(context.Background(), &proto.Rule{}, "", &rrr)
	assert.NotNil(t, eee)

	itemRes0 := ItemResult{
		Label: string(biz.REVIEW),
	}
	assert.Equal(t, itemRes0.GetScore(), float32(0.333333))

	itemRes1 := ItemResult{
		Label: string(biz.PASS),
	}
	assert.Equal(t, itemRes1.GetScore(), float32(0.999999))

	itemRes2 := ItemResult{
		Label: string(biz.BLOCK),
	}
	assert.Equal(t, itemRes2.GetScore(), float32(0.999999))

	itemRes3 := ItemResult{
		Label: string(biz.BLOCK),
		Score: 0.654321,
	}
	assert.Equal(t, itemRes3.GetScore(), float32(0.654321))

	itemRes4 := ItemResult{
		Label: string(biz.BLOCK),
		Faces: []struct {
			BoundingBox struct {
				Pts   [][2]int `json:"pts"`
				Score float32  `json:"score"`
			} `json:"bounding_box"`
			Faces []struct {
				ID    string  `json:"id,omitempty"`
				Name  string  `json:"name,omitempty"`
				Score float32 `json:"score"`
				Group string  `json:"group,omitempty"`
			} `json:"faces,omitempty"`
		}{
			struct {
				BoundingBox struct {
					Pts   [][2]int `json:"pts"`
					Score float32  `json:"score"`
				} `json:"bounding_box"`
				Faces []struct {
					ID    string  `json:"id,omitempty"`
					Name  string  `json:"name,omitempty"`
					Score float32 `json:"score"`
					Group string  `json:"group,omitempty"`
				} `json:"faces,omitempty"`
			}{
				Faces: []struct {
					ID    string  `json:"id,omitempty"`
					Name  string  `json:"name,omitempty"`
					Score float32 `json:"score"`
					Group string  `json:"group,omitempty"`
				}{
					struct {
						ID    string  `json:"id,omitempty"`
						Name  string  `json:"name,omitempty"`
						Score float32 `json:"score"`
						Group string  `json:"group,omitempty"`
					}{
						Name:  "abc",
						Score: 0.111,
					},
					struct {
						ID    string  `json:"id,omitempty"`
						Name  string  `json:"name,omitempty"`
						Score float32 `json:"score"`
						Group string  `json:"group,omitempty"`
					}{
						Name:  "abc",
						Score: 0.222,
					},
				},
			},
		},
	}
	assert.Equal(t, itemRes4.GetScore(), float32(0.222))

	videoCutMap := make(map[int64]client.VideoCut)
	fillVideoCuts(context.Background(), videoCutMap, "pulp", []biz.SegmentResult{
		biz.SegmentResult{
			Cuts: []biz.CutResult{
				biz.CutResult{
					Offset: 11,
					URI:    "11",
				},
			},
		},
		biz.SegmentResult{
			Cuts: []biz.CutResult{
				biz.CutResult{
					Offset: 22,
					URI:    "22",
				},
				biz.CutResult{
					Offset: 44,
					URI:    "44",
				},
			},
		},
		biz.SegmentResult{
			Cuts: []biz.CutResult{
				biz.CutResult{
					Offset: 33,
					URI:    "33",
				},
			},
		},
	})

	assert.Equal(t, 4, len(videoCutMap))

	vcuts0 := getSortedCuts(videoCutMap)
	assert.Equal(t, 4, len(vcuts0))

	offsets := Offsets{
		offsets: make([]int64, 0, len(videoCutMap)),
	}
	for k := range videoCutMap {
		offsets.offsets = append(offsets.offsets, k)
	}
	sort.Sort(offsets)

	vcuts := make([]client.VideoCut, 0, len(videoCutMap))
	for _, offset := range offsets.offsets {
		vcuts = append(vcuts, videoCutMap[offset])
	}
	assert.Equal(t, 4, len(vcuts))
}
