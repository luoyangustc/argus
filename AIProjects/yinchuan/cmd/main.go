package main

import (
	ocrclassify "qiniu.com/argus/AIProjects/yinchuan/ocr/image_sync"
	"qiniu.com/argus/service/scenario"
	SCENARIO "qiniu.com/argus/service/scenario/image_sync"
	ocrbankcard "qiniu.com/argus/service/service/image/ocrbankcard/image_sync"
	ocridcard "qiniu.com/argus/service/service/image/ocridcard/image_sync"
	ocrtext "qiniu.com/argus/service/service/image/ocrtext/image_sync"
)

func main() {
	ss := SCENARIO.New()
	scenario.Main(ss, func() error {
		registers := make([]func(s interface{}), 0)
		registers = append(registers, ocridcard.Import("qiniu.com/argus/service/service/image/ocridcard/image_sync"))
		registers = append(registers, ocrtext.Import("qiniu.com/argus/service/service/image/ocrtext/image_sync"))
		registers = append(registers, ocrbankcard.Import("qiniu.com/argus/service/service/image/bankcard/image_sync"))
		registers = append(registers, ocrclassify.Import("qiniu.com/argus/AIProjects/yinchuan/ocr/image_sync"))
		for _, register := range registers {
			register(ss)
		}
		return nil
	})
}
