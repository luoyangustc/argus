package main

import (
	"encoding/json"
	"fmt"

	_ "qiniu.com/argus/licence"
)

// AVA_LICENCE=ava_licence.darwin go run -tags=ava_licence example.go

func main() {
	fmt.Println("run ok")
	// fmt.Println(jsonHelper(feature.LoadFeature()))
}

// nolint
func jsonHelper(data interface{}) string {
	buf, err := json.Marshal(data)
	if err != nil {
		return err.Error()
	}
	return string(buf)
}
