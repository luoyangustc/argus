package main

import (
	OPS "qiniu.com/argus/AIProjects/ducha/ops"
	MAIN "qiniu.com/argus/cmd/argus-live-private"
)

func main() {
	{
		OPS.RegisterDucha()
	}
	MAIN.Main()
}
