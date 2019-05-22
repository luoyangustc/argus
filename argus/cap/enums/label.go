package enums

type LabelModeType string

const (
	ModePulp                 LabelModeType = "mode_pulp"
	ModePolitician           LabelModeType = "mode_politician"
	ModeTerror               LabelModeType = "mode_terror"
	ModePoliticianPulp       LabelModeType = "mode_politician_pulp"
	ModePulpTerror           LabelModeType = "mode_pulp_terror"
	ModePoliticianTerror     LabelModeType = "mode_politician_terror"
	ModePoliticianPulpTerror LabelModeType = "mode_politician_pulp_terror"
)
