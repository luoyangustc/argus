package main

import (
	SS "qiniu.com/argus/argus/simple_service"
	argus "qiniu.com/argus/utility"
)

func main() {
	SS.Main("argus-faceg", "v1", argus.NewFaceGroupService())
}
