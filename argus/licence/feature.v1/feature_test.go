package feature

import (
	"runtime"
	"testing"
)

func TestLoadFeature(t *testing.T) {
	f := LoadFeature()
	if f.Version != FeatureVersion {
		t.Fatal("Version not match", f.Version, FeatureVersion)
	}
	if f.OS != runtime.GOOS {
		t.Fatal("OS not match", f.OS, runtime.GOOS)
	}
	if f.MemorySize == 0 {
		t.Fatal("MemorySize not ok", f.Version)
	}
	if f.CPUNum == 0 {
		t.Fatal("CPUNum not ok", f.CPUNum)
	}
}
