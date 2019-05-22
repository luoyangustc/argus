package enums

//  Scene define
type Scene string

const (
	ScenePulp       Scene = "pulp"
	SceneTerror     Scene = "terror"
	ScenePolitician Scene = "politician"
)

var (
	Scenes = []Scene{
		ScenePulp, SceneTerror, ScenePolitician,
	}
)

func (this Scene) IsValid() bool {
	switch this {
	case "pulp", "terror", "politician":
		return true
	default:
		return false
	}
}
