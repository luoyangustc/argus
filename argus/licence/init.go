// +build ava_licence

package licence

import (
	"fmt"
	"os"
	"time"

	feature "qiniu.com/argus/licence/feature.v1"
)

var (
	_defaultLicenceFileName = "ava_licence"
)

func init() {
	fname, ok := os.LookupEnv("AVA_LICENCE")
	if ok {
		_defaultLicenceFileName = fname
	}
	lic, err := LoadLicence(_defaultLicenceFileName)
	if err != nil {
		fmt.Println("load licence error", err)
		os.Exit(1)
	}
	f := feature.LoadFeature()
	check(lic, f)
	// start background check
	go func() {
		for {
			select {
			case <-time.After(time.Minute):
				check(lic, f)
			}
		}
	}()
}

func check(lic Licence, f feature.Feature) {
	result, err := lic.Match(f)
	if err != nil || result != VALID {
		fmt.Println("feature licence match failed")
		fmt.Println("detail:\n", err)
		os.Exit(2)
	}
}
