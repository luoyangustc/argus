package main

import (
	"qiniu.com/argus/service/scenario"
	SCENARIO "qiniu.com/argus/service/scenario/image_sync"

	ads "qiniu.com/argus/service/service/image/ads/image_sync"
	censor "qiniu.com/argus/service/service/image/censor/image_sync"
	face "qiniu.com/argus/service/service/image/face/image_sync"
	objectdetect "qiniu.com/argus/service/service/image/objectdetect/image_sync"
	ocridcard "qiniu.com/argus/service/service/image/ocridcard/image_sync"
	ocrtext "qiniu.com/argus/service/service/image/ocrtext/image_sync"
	ocrvat "qiniu.com/argus/service/service/image/ocrvat/image_sync"
	politician "qiniu.com/argus/service/service/image/politician/image_sync"
	pulp "qiniu.com/argus/service/service/image/pulp/image_sync"
	scene "qiniu.com/argus/service/service/image/scene/image_sync"
	terror "qiniu.com/argus/service/service/image/terror/image_sync"
	terror_complex "qiniu.com/argus/service/service/image/terror_complex/image_sync"
)

func main() {
	ss := SCENARIO.New()
	scenario.Main(ss, func() error {
		registers := make([]func(s interface{}), 0)
		registers = append(registers, pulp.Import("qiniu.com/argus/service/service/image/pulp/image_sync"))
		registers = append(registers, terror.Import("qiniu.com/argus/service/service/image/terror/image_sync"))
		registers = append(registers, ads.Import("qiniu.com/argus/service/service/image/ads/image_sync"))
		registers = append(registers, terror_complex.Import("qiniu.com/argus/service/service/image/terror_complex/image_sync"))
		registers = append(registers, politician.Import("qiniu.com/argus/service/service/image/politician/image_sync"))
		registers = append(registers, face.Import("qiniu.com/argus/service/service/image/face/image_sync"))
		registers = append(registers, censor.Import("qiniu.com/argus/service/service/image/censor/image_sync"))
		registers = append(registers, ocridcard.Import("qiniu.com/argus/service/service/image/ocridcard/image_sync"))
		registers = append(registers, ocrtext.Import("qiniu.com/argus/service/service/image/ocrtext/image_sync"))
		registers = append(registers, objectdetect.Import("qiniu.com/argus/service/service/image/objectdetect/image_sync"))
		registers = append(registers, scene.Import("qiniu.com/argus/service/service/image/scene/image_sync"))
		registers = append(registers, ocrvat.Import("qiniu.com/argus/service/service/image/ocrvat/image_sync"))
		for _, register := range registers {
			register(ss)
		}
		return nil
	})
}
