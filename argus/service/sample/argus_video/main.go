package main

import (
	"qiniu.com/argus/service/scenario"
	SCENARIO "qiniu.com/argus/service/scenario/video"
	face_search "qiniu.com/argus/service/service/video/vod/face_search/video"
	politician "qiniu.com/argus/service/service/video/vod/politician/video"
	pulp "qiniu.com/argus/service/service/video/vod/pulp/video"
	terror "qiniu.com/argus/service/service/video/vod/terror/video"
	terror_complex "qiniu.com/argus/service/service/video/vod/terror_complex/video"
)

func main() {
	ss := SCENARIO.New()
	scenario.Main(ss, func() error {
		registers := make([]func(s interface{}), 0)
		registers = append(registers, terror.Import("qiniu.com/argus/service/service/video/vod/terror/video"))
		registers = append(registers, terror_complex.Import("qiniu.com/argus/service/service/video/vod/terror_complex/video"))
		registers = append(registers, politician.Import("qiniu.com/argus/service/service/video/vod/politician/video"))
		registers = append(registers, pulp.Import("qiniu.com/argus/service/service/video/vod/pulp/video"))
		registers = append(registers, face_search.Import("qiniu.com/argus/service/service/video/vod/face_search/video"))
		for _, register := range registers {
			register(ss)
		}
		return nil
	})
}
