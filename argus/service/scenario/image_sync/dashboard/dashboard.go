package dashboard

type Dashboard []byte

func Load() map[string]Dashboard {
	return map[string]Dashboard{
		"ovewview": overviewDashboard,
		"detail":   detailDashboard,
	}
}
