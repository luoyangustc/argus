package tensor

import (
	"context"
)

func Classify(ct context.Context, reqStr string) (ret interface{}, err error) {
	// mock classify interface, for unit test
	return
}

func ClassifierInit(ct context.Context, files map[string]string, batchSize int) (err error) {
	// mock classify init interface, for unit test
	return
}

func DetecterInit(ct context.Context, files map[string]string, batchSize int) (err error) {
	// mock detect init interface, for unit test
	return
}

func Detect(ct context.Context, reqStr string) (ret interface{}, err error) {
	// mock detect interface, for unit test
	return
}
