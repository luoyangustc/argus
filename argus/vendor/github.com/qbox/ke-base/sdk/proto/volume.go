package proto

import (
	"time"
)

type StorageTypeString string

const (
	StorageTypeCeph StorageTypeString = "ceph"
)

type StorageType struct {
	Type StorageTypeString `json:"type"`
	Desc string            `json:"desc"`
	IOPS string            `json:"iops"`
}

// StorageTypes 系统指定的多种存储类型
var StorageTypes = []*StorageType{
	&StorageType{
		Type: StorageTypeCeph,
		Desc: "Desc",
		IOPS: "IPOS",
	},
}

// VolumeStatus ...
type VolumeStatus string

const (
	VolumeStatusCreating VolumeStatus = "creating"
	VolumeStatusSuccess  VolumeStatus = "success"
	VolumeStatusError    VolumeStatus = "error"
)

// VolumeFSType ...
type VolumeFSType string

const (
	VolumeFSTypeExt4 VolumeFSType = "ext4"
)

// Volume ...
type Volume struct {
	Name        string            `json:"name"`
	Size        int64             `json:"size"`
	Status      VolumeStatus      `json:"status"`
	StorageType StorageTypeString `json:"storageType"`
	FSType      VolumeFSType      `json:"fsType"`
	CTime       time.Time         `json:"creationTime"`
}
