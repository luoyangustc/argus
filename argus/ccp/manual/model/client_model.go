package model

import (
	"encoding/json"
	"fmt"
	"net/url"

	"qiniu.com/argus/censor/biz"

	ENUMS "qiniu.com/argus/cap/enums"
	MODEL "qiniu.com/argus/cap/model"
)

const CAPSCORE = 0.999

type BatchTasksReq struct {
	ID     string            `json:"id"`  //TaskId
	URI    string            `json:"uri"` //Task url
	Labels []MODEL.LabelInfo `json:"label"`
	//	Frames interface{} `json:"frames", omitempty` //TODO: video
}

func FromImageBcpResultModel(uid uint32, taskId, uri string, req *BcpResultModel) *BatchTasksReq {
	var (
		resp = BatchTasksReq{
			ID:     taskId,
			URI:    convQiniuUrl(uid, uri),
			Labels: make([]MODEL.LabelInfo, 0),
		}
		bcpResult = biz.CensorResponse{}
		taskType  = ENUMS.LableClassification
	)

	err := json.Unmarshal(req.Result, &bcpResult)
	if err != nil {
		return &resp
	}

	//如果有政治人物，Type全部更新为"detection"
	for k := range bcpResult.Scenes {
		if string(k) == string(biz.POLITICIAN) {
			taskType = ENUMS.LabelDetection
			break
		}
	}

	for k, v := range bcpResult.Scenes {
		imgResp := biz.ImageSceneResponse{}
		_ = convByJson(v, &imgResp)

		switch string(k) {
		case string(biz.PULP):
			pulpInfo := biz.PulpResult{}
			err := convByJson(imgResp.Result, &pulpInfo)
			if err == nil {
				labelInfo := MODEL.LabelInfo{
					Name: string(biz.PULP),
					Type: taskType,
					Data: make([]interface{}, 0),
				}
				lData := MODEL.LabelData{
					Class: pulpInfo.Label,
					Score: pulpInfo.Score,
				}

				labelInfo.Data = append(labelInfo.Data, lData)
				resp.Labels = append(resp.Labels, labelInfo)
			}
		case string(biz.TERROR):
			tInfo := biz.TerrorResult{}
			err := convByJson(imgResp.Result, &tInfo)
			if err == nil {
				labelInfo := MODEL.LabelInfo{
					Name: string(biz.TERROR),
					Type: taskType,
					Data: make([]interface{}, 0),
				}
				lData := MODEL.LabelData{
					Class: tInfo.Label,
					Score: tInfo.Score,
				}
				labelInfo.Data = append(labelInfo.Data, lData)
				resp.Labels = append(resp.Labels, labelInfo)
			}
		case string(biz.POLITICIAN):
			pInfo := biz.PoliticianResult{}
			err := convByJson(imgResp.Result, &pInfo)
			if err == nil {
				labelInfo := MODEL.LabelInfo{
					Name: string(biz.POLITICIAN),
					Type: taskType,
					Data: make([]interface{}, 0),
				}
				{
					//添加Data
					lData := MODEL.LabelPoliticianData{
						Class: pInfo.Label,
						Faces: make([]struct {
							BoundingBox struct {
								Pts   [][2]int `json:"pts"`
								Score float32  `json:"score"`
							} `json:"bounding_box"`
							Faces []struct {
								ID     string  `json:"id,omitempty"`
								Name   string  `json:"name,omitempty"`
								Score  float32 `json:"score"`
								Group  string  `json:"group,omitempty"`
								Sample *struct {
									URL string   `json:"url"`
									Pts [][2]int `json:"pts"`
								} `json:"sample,omitempty"`
							} `json:"faces,omitempty"`
						}, 0),
					}

					for _, v := range pInfo.Faces {
						face := struct {
							BoundingBox struct {
								Pts   [][2]int `json:"pts"`
								Score float32  `json:"score"`
							} `json:"bounding_box"`
							Faces []struct {
								ID     string  `json:"id,omitempty"`
								Name   string  `json:"name,omitempty"`
								Score  float32 `json:"score"`
								Group  string  `json:"group,omitempty"`
								Sample *struct {
									URL string   `json:"url"`
									Pts [][2]int `json:"pts"`
								} `json:"sample,omitempty"`
							} `json:"faces,omitempty"`
						}{
							BoundingBox: v.BoundingBox,
							Faces:       v.Faces,
						}

						lData.Faces = append(lData.Faces, face)
					}

					labelInfo.Data = append(labelInfo.Data, lData)
					resp.Labels = append(resp.Labels, labelInfo)
				}
			}
		}
	}

	return &resp
}

func AddInfoForErr(uid uint32, taskId, uri string, scenes []string) *BatchTasksReq {
	var (
		resp = BatchTasksReq{
			ID:     taskId,
			URI:    convQiniuUrl(uid, uri),
			Labels: make([]MODEL.LabelInfo, 0),
		}
		taskType = ENUMS.LableClassification
	)

	//如果有政治人物，Type全部更新为"detection"
	for k := range scenes {
		if string(k) == string(biz.POLITICIAN) {
			taskType = ENUMS.LabelDetection
			break
		}
	}

	for _, v := range scenes {
		switch v {
		case string(biz.PULP):
			labelInfo := MODEL.LabelInfo{
				Name: string(biz.PULP),
				Type: taskType,
				Data: make([]interface{}, 0),
			}
			lData := MODEL.LabelData{
				Class: biz.PULP_NORMAL,
				Score: CAPSCORE,
			}
			labelInfo.Data = append(labelInfo.Data, lData)
			resp.Labels = append(resp.Labels, labelInfo)

		case string(biz.TERROR):
			labelInfo := MODEL.LabelInfo{
				Name: string(biz.TERROR),
				Type: taskType,
				Data: make([]interface{}, 0),
			}
			lData := MODEL.LabelData{
				Class: biz.TERROR_NORMAL,
				Score: CAPSCORE,
			}
			labelInfo.Data = append(labelInfo.Data, lData)
			resp.Labels = append(resp.Labels, labelInfo)

		case string(biz.POLITICIAN):
			labelInfo := MODEL.LabelInfo{
				Name: string(biz.POLITICIAN),
				Type: taskType,
				Data: make([]interface{}, 0),
			}

			pData := MODEL.LabelPoliticianData{
				Class: string(biz.POLITICIAN),
			}
			labelInfo.Data = append(labelInfo.Data, pData)
			resp.Labels = append(resp.Labels, labelInfo)
		}
	}

	return &resp
}

func convByJson(src interface{}, dest interface{}) error {

	tmpbs, err := json.Marshal(src)
	if err != nil {
		return err
	}

	return json.Unmarshal(tmpbs, dest)
}

func convQiniuUrl(uid uint32, qiniuURL string) string {
	u, err := url.Parse(qiniuURL)
	if err == nil && u.Scheme == "qiniu" {
		return fmt.Sprintf("%s://%d@%s%s", u.Scheme, uid, u.Host, u.Path)
	}
	return qiniuURL
}
