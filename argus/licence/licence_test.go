package licence

import (
	"testing"
	"time"

	feature "qiniu.com/argus/licence/feature.v1"
)

func TestExist(t *testing.T) {
	if Exist(nil) {
		t.Fatal("fail")
	}
	if !Exist(t) {
		t.Fatal("fail")
	}
	nilString := (*string)(nil)
	if Exist(nilString) {
		t.Fatal("fail")
	}
}

func TestLessEqual(t *testing.T) {
	if !LessEqual(0, 1) {
		t.Fatal("fail")
	}
	if !LessEqual(1, 1) {
		t.Fatal("fail")
	}
	if !LessEqual(1, 2) {
		t.Fatal("fail")
	}
}

func TestIn(t *testing.T) {
	if !In("a", []string{}) {
		t.Fatal("fail")
	}
	if !In("a", []string{"a"}) {
		t.Fatal("fail")
	}
	if In("a", []string{"b"}) {
		t.Fatal("fail")
	}
	if !In("a", []string{"a", "b"}) {
		t.Fatal("fail")
	}
}

func TestAllIn(t *testing.T) {
	if !AllIn([]string{"a"}, []string{}) {
		t.Fatal("fail")
	}
	if !AllIn([]string{"a"}, []string{"a"}) {
		t.Fatal("fail")
	}
	if AllIn([]string{"a"}, []string{"b"}) {
		t.Fatal("fail")
	}
	if !AllIn([]string{"a"}, []string{"a", "b"}) {
		t.Fatal("fail")
	}
	if !AllIn([]string{"a", "b"}, []string{"a", "b"}) {
		t.Fatal("fail")
	}
}

func TestHasIn(t *testing.T) {
	if !HasIn([]string{"a"}, []string{}) {
		t.Fatal("fail")
	}
	if !HasIn([]string{"a"}, []string{"a"}) {
		t.Fatal("fail")
	}
	if HasIn([]string{"a"}, []string{"b"}) {
		t.Fatal("fail")
	}
	if !HasIn([]string{"a"}, []string{"a", "b"}) {
		t.Fatal("fail")
	}
	if !HasIn([]string{"a", "c"}, []string{"a", "b"}) {
		t.Fatal("fail")
	}
}
func TestMatch(t *testing.T) {
	lc := Policy{}
	f := feature.Feature{}

	if value, _ := lc.Match(f); value != INVALID {
		t.Fatal("nill Version accept")
	}
	Version := feature.FeatureVersion + "test"

	lc.Version = &Version
	f.Version = feature.FeatureVersion
	if value, _ := lc.Match(f); value != INVALID {
		t.Fatal("nill Version accept")
	}
	Version = feature.FeatureVersion

	if value, _ := lc.Match(f); value != INVALID {
		t.Fatal("nill Expires accept")
	}
	Expires := time.Now()
	lc.Expires = &Expires

	lc.OS = feature.GetOS()
	if value, _ := lc.Match(f); value != INVALID {
		t.Fatal("error OS accept")
	}
	f.OS = feature.GetOS()

	lc.AppName = "test"
	if value, _ := lc.Match(f); value != INVALID {
		t.Fatal("error AppName accept", value)
	}
	f.AppName = lc.AppName

	lc.AppMd5 = feature.GetAppMd5()
	if value, _ := lc.Match(f); value != INVALID {
		t.Fatal("error AppMd5 accept")
	}
	f.AppMd5 = feature.GetAppMd5()

	lc.AppVersion = "vtest"
	if value, _ := lc.Match(f); value != INVALID {
		t.Fatal("error AppVersion accept")
	}
	f.AppVersion = lc.AppVersion

	lc.DataMd5 = []string{"data_test"}
	f.DataMd5 = []string{"data_test22222"}
	if value, _ := lc.Match(f); value != INVALID {
		t.Fatal("error DataMd5 accept")
	}
	f.DataMd5 = lc.DataMd5

	lc.MemorySize = feature.GetTotalMemory() - 1
	f.MemorySize = feature.GetTotalMemory()
	if value, _ := lc.Match(f); value != INVALID {
		t.Fatal("error MemorySize accept", lc.MemorySize, f.MemorySize)
	}
	lc.MemorySize = feature.GetTotalMemory()

	lc.CPUNum = feature.GetCPUNum() - 1
	f.CPUNum = feature.GetCPUNum()
	if value, _ := lc.Match(f); value != INVALID {
		t.Fatal("error CPUNum accept")
	}
	lc.CPUNum = feature.GetCPUNum()

	f.SystemUUID = "abcde"
	lc.SystemUUID = []string{"abcde2"}
	if value, _ := lc.Match(f); value != INVALID {
		t.Fatal("error SystemUUID accept")
	}
	lc.SystemUUID = []string{"abcde", "abcde2"}

	f.GPUUUID = []string{"abcde"}
	lc.GPUUUID = []string{"abcde2"}
	if value, _ := lc.Match(f); value != INVALID {
		t.Fatal("error GPUUUID accept")
	}
	lc.GPUUUID = []string{"abcde", "abcde2"}

	f.MacAddress = []string{"abcde"}
	lc.MacAddress = []string{"abcde2"}
	if value, _ := lc.Match(f); value != INVALID {
		t.Fatal("error MacAddress accept")
	}
	lc.MacAddress = []string{"abcde", "abcde2"}

	*lc.Expires = time.Now().Add(-time.Minute)
	if value, err := lc.Match(f); value != EXPIRED {
		t.Fatal("error Expires accept", err)
	}
	*lc.Expires = time.Now().Add(time.Minute)

	if value, _ := lc.Match(f); value == INVALID {
		t.Fatal("error case")
	}
}
