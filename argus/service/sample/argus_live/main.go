package main

import (
	"qiniu.com/argus/service/scenario"
	SCENARIO "qiniu.com/argus/service/scenario/video"
	foo "qiniu.com/argus/service/service/video/live/foo/video"
)

func main() {
	ss := SCENARIO.New()
	scenario.Main(ss, func() error {
		registers := make([]func(s interface{}), 0)
		registers = append(registers, foo.Import("qiniu.com/argus/service/service/video/live/foo/video"))
		for _, register := range registers {
			register(ss)
		}
		return nil
	})
}
