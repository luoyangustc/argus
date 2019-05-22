package version

import (
	"fmt"
	"os"
)

var version string = "develop"

func init() {
	if len(os.Args) > 1 && os.Args[1] == "-version" {
		fmt.Println(version)
		os.Exit(0)
	}
}

func Version() string {
	return version
}
