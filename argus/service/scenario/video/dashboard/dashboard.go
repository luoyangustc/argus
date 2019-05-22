package dashboard

type Dashboard []byte

func Load() map[string]Dashboard {
	return map[string]Dashboard{
		"video_ovewview": videoOverviewDashboard,
	}
}
