package feature

import (
	"github.com/mindprince/gonvml"
)

func GetGpuUUID() (value []string) {
	err := gonvml.Initialize()
	if err != nil {
		return
	}
	defer func() { _ = gonvml.Shutdown() }()

	n, err := gonvml.DeviceCount()
	if err != nil {
		return
	}
	for i := uint(0); i < n; i++ {
		d, err := gonvml.DeviceHandleByIndex(i)
		if err != nil {
			continue
		}
		uid, err := d.UUID()
		if err != nil {
			continue
		}
		if uid != "" {
			value = append(value, uid)
		}
	}

	return
}
