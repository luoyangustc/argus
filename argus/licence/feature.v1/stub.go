// +build !linux

package feature

// GetGpuUUID ...
//
func GetGpuUUID() []string {
	return nil
}

// GetSystemUUID ...
//
func GetSystemUUID() string {
	return ""
}
