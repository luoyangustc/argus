package biz

type ImageSceneResponse struct { // TODO
	Suggestion Suggestion  `json:"suggestion"`
	Result     interface{} `json:"result,omitempty"`
}

////////////////////////////////////////////////////////////////////////////////

func ParseImagePulpResp(
	resp0 ImagePulpResp, params PulpThreshold,
) (resp ImageSceneResponse) {
	resp.Suggestion, resp.Result = ParsePulp(resp0, params)
	return
}

func ParseImageTerrorResp(
	resp0 ImageTerrorResp, params TerrorThreshold,
) (resp ImageSceneResponse) {
	resp.Suggestion, resp.Result = ParseTerror(resp0, params)
	return
}

func ParseImagePoliticianResp(
	resp0 ImagePoliticianResp, params PoliticianThreshold,
) (resp ImageSceneResponse) {
	resp.Suggestion, resp.Result = ParsePolitician(resp0, params)
	return
}
