package video

import (
	scenario "qiniu.com/argus/service/scenario/video"
	"qiniu.com/argus/service/service/video/live/foo"
	live "qiniu.com/argus/service/service/video/live/video"
)

const (
	VERSION = "1.0.0"
)

func Import(serviceID string) func(interface{}) {
	return func(s0 interface{}) {
		s := s0.(scenario.VideoService)
		Init(s, serviceID)
	}
}

func Init(s scenario.VideoService, serviceID string) {

	live.GetSet(s, live.ID).
		RegisterOP(scenario.ServiceInfo{ID: serviceID, Version: VERSION},
			"foo", nil, func() interface{} {
				return foo.NewOP()
			})

}
