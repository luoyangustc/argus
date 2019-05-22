package cuckoofilter

import (
	"testing"
)

func TestIndexAndFP(t *testing.T) {
	data := []byte("seif")
	i1, i2, fp := getIndicesAndFingerprint(data, 1024)
	i11 := getAltIndex(fp, i2, 1024)
	i22 := getAltIndex(fp, i1, 1024)
	if i1 != i11 {
		t.Errorf("Expected i1 == i11, instead %d != %d", i1, i11)
	}
	if i2 != i22 {
		t.Errorf("Expected i2 == i22, instead %d != %d", i2, i22)
	}
}
