package main

import (
	MAIN "qiniu.com/argus/cmd/argus-live-private"
	OPS "qiniu.com/argus/video/ops"
)

func main() {
	{
		OPS.RegisterPulp()
		OPS.RegisterTerror()
		OPS.RegisterPolitician()
		OPS.RegisterFaceDetect()
		OPS.RegisterFaceGroupSearch()
		OPS.RegisterImageLabel()

		OPS.RegisterTerrorClassify()
		OPS.RegisterTerrorDetect()
		OPS.RegisterDetection()
		OPS.RegisterMagicearTrophy()
		OPS.RegisterFaceGroupPrivateSearch()
	}
	MAIN.Main()
}
