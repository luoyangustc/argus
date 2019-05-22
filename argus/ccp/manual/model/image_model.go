package model

import (
	"encoding/json"

	xlog "github.com/qiniu/xlog.v1"

	MODEL "qiniu.com/argus/cap/model"
	"qiniu.com/argus/ccp/manual/enums"
	"qiniu.com/argus/censor/biz"
)

type Scene string
type Suggestion string

type BcpResultModel struct {
	Code     int             `json:"code"`
	Mimetype string          `json:"mimetype"`
	Error    string          `json:"error,omitempty"`
	Result   json.RawMessage `json:"result,omitempty"` // biz.CensorResponse
}

func FromCAPResult(req *MODEL.TaskResult) *BcpResultModel {
	resp := BcpResultModel{
		Code:     200,
		Mimetype: "image", //TODO: -> Video
	}
	censorResp := biz.CensorResponse{
		Code:   200,
		Scenes: make(map[biz.Scene]interface{}),
	}
	for _, label := range req.Labels {
		switch label.Name {
		case string(biz.PULP):
			pData := []MODEL.LabelData{}
			err := convByJson(label.Data, &pData)
			if err != nil {
				continue
			}
			if len(pData) > 0 {
				pulpResult := biz.PulpResult{
					Label: pData[0].Class,
					Score: CAPSCORE,
				}

				sugges := convSuggestion(biz.PULP, pulpResult.Label)
				censorResp.Scenes[biz.PULP] = biz.ImageSceneResponse{
					Suggestion: sugges,
					Result:     pulpResult,
				}
				censorResp.Suggestion = updateSuggestion(censorResp.Suggestion, sugges)
			}

		case string(biz.TERROR):
			tData := []MODEL.LabelData{}
			err := convByJson(label.Data, &tData)
			if err != nil {
				continue
			}
			if len(tData) > 0 {
				terrorResult := biz.TerrorResult{
					Label: tData[0].Class,
					Score: CAPSCORE,
				}
				sugges := convSuggestion(biz.TERROR, terrorResult.Label)
				censorResp.Scenes[biz.TERROR] = biz.ImageSceneResponse{
					Suggestion: sugges,
					Result:     terrorResult,
				}
				censorResp.Suggestion = updateSuggestion(censorResp.Suggestion, sugges)
			}

		case string(biz.POLITICIAN):
			poData := []MODEL.LabelPoliticianData{}
			err := convByJson(label.Data, &poData)
			if err != nil {
				continue
			}
			if len(poData) > 0 {
				poResult := biz.PoliticianResult{
					Label: poData[0].Class,
					Faces: poData[0].Faces,
				}
				sugges := biz.REVIEW
				if len(poResult.Faces) > 0 && len(poResult.Faces[0].Faces) > 0 {
					sugges = convSuggestion(biz.POLITICIAN, poResult.Faces[0].Faces[0].Group)
				}

				censorResp.Scenes[biz.POLITICIAN] = biz.ImageSceneResponse{
					Suggestion: sugges,
					Result:     poResult,
				}
				censorResp.Suggestion = updateSuggestion(censorResp.Suggestion, sugges)
			}
		}
	}

	result, err := json.Marshal(censorResp)
	if err != nil {
		xlog.Errorf("json.Marshal censorResp error: %#v", err.Error())
		return &resp
	}

	resp.Result = result
	return &resp
}

func convSuggestion(scene biz.Scene, label string) (suggestion biz.Suggestion) {
	switch scene {
	case biz.PULP:
		switch label {
		case enums.PULP_normal, enums.PULP_sexy:
			suggestion = biz.PASS
		case enums.PULP_pulp:
			suggestion = biz.BLOCK
		}
	case biz.TERROR:
		switch label {
		case enums.TERROR_normal:
			suggestion = biz.PASS
		case enums.TERROR_beheaded, enums.TERROR_illegal_flag, enums.TERROR_guns:
			suggestion = biz.BLOCK
		default:
			suggestion = biz.REVIEW
		}
	case biz.POLITICIAN:
		switch label {
		case enums.POLITICIAN_normal:
			suggestion = biz.PASS
		case enums.POLITICIAN_affairs_official_gov, enums.POLITICIAN_affairs_official_ent:
			suggestion = biz.BLOCK
		default:
			suggestion = biz.REVIEW
		}
	}

	return
}

func updateSuggestion(oldV biz.Suggestion, newV biz.Suggestion) biz.Suggestion {
	if oldV == biz.BLOCK {
		return oldV
	}
	if newV == biz.BLOCK {
		return newV
	}
	if oldV == biz.REVIEW {
		return oldV
	}
	if newV == biz.REVIEW {
		return newV
	}
	return newV
}
