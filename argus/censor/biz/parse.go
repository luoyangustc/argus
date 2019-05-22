package biz

import "qiniu.com/argus/utility/censor"

type PulpThreshold struct {
	Pulp   *float32 `json:"pulp,omitempty"`
	Sexy   *float32 `json:"sexy,omitempty"`
	Normal *float32 `json:"normal,omitempty"`
}
type ImagePulpResp censor.PulpResult

func ParsePulp(resp ImagePulpResp, thresholds PulpThreshold,
) (suggestion Suggestion, ret PulpResult) {
	suggestion = PASS
	switch resp.Label {
	case 0:
		ret.Label, ret.Score = PULP_PULP, resp.Score
		suggestion = co(thresholds.Pulp == nil,
			func() interface{} { return co(resp.Review, v(REVIEW), v(BLOCK)) },
			func() interface{} { return co(resp.Score < *thresholds.Pulp, v(REVIEW), v(BLOCK)) },
		).(Suggestion)
	case 1:
		ret.Label, ret.Score = PULP_SEXY, resp.Score
		suggestion = co(thresholds.Sexy == nil, v(REVIEW),
			func() interface{} { return co(resp.Score >= *thresholds.Sexy, v(BLOCK), v(REVIEW)) },
		).(Suggestion)
	case 2:
		ret.Label, ret.Score = PULP_NORMAL, resp.Score
		suggestion = co(thresholds.Normal == nil, v(PASS),
			func() interface{} { return co(resp.Score < *thresholds.Normal, v(REVIEW), v(PASS)) },
		).(Suggestion)
	}
	return
}

type TerrorThreshold struct {
	Terror *float32 `json:"terror"`
	Normal *float32 `json:"normal"`
}
type ImageTerrorResp censor.TerrorResult

func ParseTerror(resp ImageTerrorResp, thresholds TerrorThreshold,
) (suggestion Suggestion, ret TerrorResult) {
	suggestion = PASS
	switch resp.Label {
	case 0:
		ret.Label, ret.Score = TERROR_NORMAL, resp.Score
		suggestion = co(thresholds.Normal == nil, v(PASS),
			func() interface{} { return co(resp.Score < *thresholds.Normal, v(REVIEW), v(PASS)) },
		).(Suggestion)
	case 1:
		ret.Label, ret.Score = co(resp.Class == "", v(TERROR_TERROR), v(resp.Class)).(string), resp.Score
		suggestion = co(thresholds.Terror == nil,
			func() interface{} { return co(resp.Review, v(REVIEW), v(BLOCK)) },
			func() interface{} { return co(resp.Score >= *thresholds.Terror, v(BLOCK), v(REVIEW)) },
		).(Suggestion)
	}
	return
}

type PoliticianThreshold struct{}
type ImagePoliticianResp censor.FaceSearchResult

func ParsePolitician(resp ImagePoliticianResp, params PoliticianThreshold,
) (suggestion Suggestion, ret PoliticianResult) {
	suggestion = PASS
	if len(resp.Detections) == 0 {
		ret.Label = POLITICIAN_NORMAL
		return
	}
	ret.Label = POLITICIAN_FACE
	for _, det := range resp.Detections {
		face := struct {
			BoundingBox struct {
				Pts   [][2]int `json:"pts"`
				Score float32  `json:"score"`
			} `json:"bounding_box"`
			Faces []struct {
				ID    string  `json:"id,omitempty"`
				Name  string  `json:"name,omitempty"`
				Score float32 `json:"score"`
				Group string  `json:"group,omitempty"`

				Sample *struct {
					URL string   `json:"url"`
					Pts [][2]int `json:"pts"`
				} `json:"sample,omitempty"`
			} `json:"faces,omitempty"`
		}{
			BoundingBox: struct {
				Pts   [][2]int `json:"pts"`
				Score float32  `json:"score"`
			}{
				Pts:   det.BoundingBox.Pts,
				Score: det.BoundingBox.Score,
			},
		}

		if det.Value.Name != "" {
			f := struct {
				ID    string  `json:"id,omitempty"`
				Name  string  `json:"name,omitempty"`
				Score float32 `json:"score"`
				Group string  `json:"group,omitempty"`

				Sample *struct {
					URL string   `json:"url"`
					Pts [][2]int `json:"pts"`
				} `json:"sample,omitempty"`
			}{
				Name:  det.Value.Name,
				Score: det.Value.Score,
				Group: det.Value.Group,
			}

			if det.Sample != nil {
				f.Sample = &struct {
					URL string   `json:"url"`
					Pts [][2]int `json:"pts"`
				}{
					URL: det.Sample.URL,
					Pts: det.Sample.Pts,
				}
			}

			face.Faces = append(face.Faces, f)
			ret.Label = POLITICIAN_POLITICIAN
		}

		suggestion = suggestion.Update(
			co(det.Value.Name == "", v(PASS),
				func() interface{} {
					return co(det.Value.Review, v(REVIEW),
						func() interface{} {
							switch det.Value.Group {
							case "domestic_statesman",
								"foreign_statesman":
								return REVIEW
							}
							return BLOCK
						},
					)
				},
			).(Suggestion))
		ret.Faces = append(ret.Faces, face)
	}
	return
}
