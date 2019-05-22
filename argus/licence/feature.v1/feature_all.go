package feature

import (
	// nolint
	"crypto/md5"
	"encoding/hex"
	"io/ioutil"
	"net"
	"os"
	"os/exec"
	"runtime"

	"github.com/pbnjay/memory"
)

// GetMacAddress ...
//
func GetMacAddress() (addr []string) {
	ifs, err := net.Interfaces()
	if err != nil {
		return addr
	}
	for _, v := range ifs {
		if v.HardwareAddr.String() != "" {
			addr = append(addr, v.HardwareAddr.String())
		}
	}
	return addr
}

// GetTotalMemory ...
//
func GetTotalMemory() uint64 {
	return memory.TotalMemory()
}

// GetCPUNum ...
//
func GetCPUNum() uint64 {
	return uint64(runtime.NumCPU())
}

// GetOS ...
//
func GetOS() string {
	return runtime.GOOS
}

// GetAppMd5 ...
//
func GetAppMd5() string {
	path, err := exec.LookPath(os.Args[0])
	if err != nil {
		return ""
	}
	bs, err := ioutil.ReadFile(path)
	if err != nil {
		return ""
	}
	// nolint
	sum := md5.Sum(bs)
	return hex.EncodeToString(sum[:])
}

// GetDiskUUID ...
//
func GetDiskUUID() string {
	return ""
}

// GetAppName ...
//
func GetAppName() string {
	return ""
}

// GetAppVersion ...
//
func GetAppVersion() string {
	return ""
}

// GetDataMd5sum ...
//
func GetDataMd5sum() []string {
	return nil
}
