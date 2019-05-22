package biz

type Suggestion string

const (
	PASS   Suggestion = "pass"
	BLOCK  Suggestion = "block"
	REVIEW Suggestion = "review"
)

func (suggestion Suggestion) Update(newV Suggestion) Suggestion {
	if suggestion == BLOCK {
		return suggestion
	}
	if newV == BLOCK {
		return newV
	}
	if suggestion == REVIEW {
		return suggestion
	}
	if newV == REVIEW {
		return newV
	}
	return newV
}

type Scene string

const (
	PULP       Scene = "pulp"
	TERROR     Scene = "terror"
	POLITICIAN Scene = "politician"
)

var DefaultScenes = []Scene{PULP, TERROR, POLITICIAN}

type CensorResponse struct {
	Code       int                   `json:"code,omitempty"`
	Message    string                `json:"message,omitempty"`
	Suggestion Suggestion            `json:"suggestion,omitempty"`
	Scenes     map[Scene]interface{} `json:"scenes,omitempty"`
}

////////////////////////////////////////////////////////////////////////////////

const (
	PULP_NORMAL = "normal"
	PULP_SEXY   = "sexy"
	PULP_PULP   = "pulp"
)

type PulpResult struct {
	Label string  `json:"label"`
	Score float32 `json:"score"`
}

const (
	TERROR_NORMAL = "normal"
	TERROR_TERROR = "terror"
)

type TerrorResult struct {
	Label string  `json:"label"`
	Score float32 `json:"score"`
}

const (
	POLITICIAN_NORMAL     = "normal"
	POLITICIAN_FACE       = "face"
	POLITICIAN_POLITICIAN = "politician"
)

type PoliticianResult struct {
	Label string `json:"label"`
	Faces []struct {
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
	} `json:"faces"`
}
