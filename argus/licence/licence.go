package licence

import (
	"fmt"
	"reflect"
	"sort"
	"time"

	feature "qiniu.com/argus/licence/feature.v1"
)

// VerifyResult ...
//
type VerifyResult int

const (
	INVALID = VerifyResult(0)
	VALID   = VerifyResult(1)
	EXPIRED = VerifyResult(2)
)

// Policy ...
//
type Policy struct {
	Version    *string    `json:"version"`
	Expires    *time.Time `json:"expires"`
	OS         string     `json:"os"`
	AppName    string     `json:"app"`
	AppVersion string     `json:"app_version"`
	AppMd5     string     `json:"app_md5"`
	DataMd5    []string   `json:"data_md5"`
	MemorySize uint64     `json:"memory_size"`
	CPUNum     uint64     `json:"cpu_num"`
	SystemUUID []string   `json:"system_uuid"`
	GPUUUID    []string   `json:"gpu_uuid"`
	DiskUUID   []string   `json:"disk_uuid"`
	MacAddress []string   `json:"mac_address"`
}

// Licence ...
//
type Licence []Policy

// Exist ...
//
func Exist(data interface{}) bool {
	if data == nil {
		return false
	}
	return !reflect.ValueOf(data).IsNil()
}

// Equal ...
//
func Equal(data string, data2 string) bool {
	return data2 == "" || data2 == data
}

// LessEqual ...
//
func LessEqual(data uint64, data2 uint64) bool {
	return data2 == 0 || data <= data2
}

// In ...
//
func In(data string, data2 []string) bool {
	if len(data2) == 0 {
		return true
	} else if index := sort.SearchStrings(data2, data); index != len(data2) && data2[index] == data {
		return true
	}
	return false
}

// AllIn ...
//
func AllIn(data []string, data2 []string) bool {
	if len(data2) == 0 {
		return true
	}
	for _, v := range data {
		if !In(v, data2) {
			return false
		}
	}
	return true
}

// HasIn ...
//
func HasIn(data []string, data2 []string) bool {
	if len(data2) == 0 {
		return true
	}
	for _, v := range data {
		if In(v, data2) {
			return true
		}
	}
	return false
}

// Sort ...
//
func (lc *Policy) Sort() {
	sort.Strings(lc.DataMd5)
	sort.Strings(lc.GPUUUID)
	sort.Strings(lc.SystemUUID)
	sort.Strings(lc.DiskUUID)
	sort.Strings(lc.MacAddress)
}

// Match ...
//
func (lc Policy) Match(f feature.Feature) (retv VerifyResult, err error) {
	retv = INVALID
	if !Exist(lc.Version) || !Exist(lc.Expires) {
		err = fmt.Errorf("Version or Expires empty")
		return
	} else if !Equal(f.Version, *lc.Version) {
		err = fmt.Errorf("Version")
		return
	} else if !Equal(f.OS, lc.OS) {
		err = fmt.Errorf("OS")
		return
	} else if !Equal(f.AppName, lc.AppName) {
		err = fmt.Errorf("AppName")
		return
	} else if !Equal(f.AppMd5, lc.AppMd5) {
		err = fmt.Errorf("AppMd5")
		return
	} else if !Equal(f.AppVersion, lc.AppVersion) {
		err = fmt.Errorf("AppVersion")
		return
	} else if !AllIn(f.DataMd5, lc.DataMd5) {
		err = fmt.Errorf("DataMd5")
		return
	} else if !LessEqual(f.MemorySize, lc.MemorySize) {
		err = fmt.Errorf("MemorySize")
		return
	} else if !LessEqual(f.CPUNum, lc.CPUNum) {
		err = fmt.Errorf("CPUNum")
		return
	} else if !In(f.SystemUUID, lc.SystemUUID) {
		err = fmt.Errorf("SystemUUID")
		return
	} else if !AllIn(f.GPUUUID, lc.GPUUUID) {
		err = fmt.Errorf("GPUUUID")
		return
	} else if !HasIn(f.MacAddress, lc.MacAddress) {
		err = fmt.Errorf("MacAddress")
		return
	}

	// check data
	if lc.Expires.Before(time.Now()) {
		return EXPIRED, fmt.Errorf("Expires error")
	}

	// all check pass
	return VALID, nil
}

// Match ...
//
func (lc Licence) Match(f feature.Feature) (r VerifyResult, err error) {
	r = INVALID
	msg := ""
	for index, v := range lc {
		v.Sort()
		r, err = v.Match(f)
		if err == nil && r == VALID {
			return
		}
		msg += fmt.Sprintf("Policy %d: '%v' miss match\n", index, err)
	}
	return r, fmt.Errorf(msg)
}
