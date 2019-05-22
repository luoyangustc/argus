package feature

const (
	// FeatureVersion value define
	FeatureVersion = ("v1")
)

// Feature of machine
//
type Feature struct {
	Version    string   `json:"version,omitempty"`
	OS         string   `json:"os,omitempty"`
	AppName    string   `json:"app,omitempty"`
	AppVersion string   `json:"app_version,omitempty"`
	AppMd5     string   `json:"app_md5,omitempty"`
	DataMd5    []string `json:"data_md5,omitempty"`
	MemorySize uint64   `json:"memory_size,omitempty"`
	CPUNum     uint64   `json:"cpu_num,omitempty"`
	SystemUUID string   `json:"system_uuid,omitempty"`
	DiskUUID   string   `json:"disk_uuid,omitempty"`
	GPUUUID    []string `json:"gpu_uuid,omitempty"`
	MacAddress []string `json:"mac_address,omitempty"`
}

// LoadFeature ...
//
func LoadFeature() Feature {
	return Feature{
		Version:    FeatureVersion,
		OS:         GetOS(),
		AppName:    GetAppName(),
		AppVersion: GetAppVersion(),
		AppMd5:     GetAppMd5(),
		DataMd5:    GetDataMd5sum(),
		MemorySize: GetTotalMemory(),
		CPUNum:     GetCPUNum(),
		SystemUUID: GetSystemUUID(),
		DiskUUID:   GetDiskUUID(),
		GPUUUID:    GetGpuUUID(),
		MacAddress: GetMacAddress(),
	}
}
