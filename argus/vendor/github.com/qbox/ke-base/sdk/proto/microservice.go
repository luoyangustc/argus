package proto

import (
	"time"
)

// App ...
type App struct {
	Name          string         `json:"name"`
	Project       string         `json:"projectName"`
	CTime         time.Time      `json:"creationTime"`
	MicroServices []MicroService `json:"microservices"`
}

// MicroServiceStatus ...
type MicroServicePhase string

const (
	MicroServicePhaseRunning MicroServicePhase = "running"
	MicroServicePhasePending MicroServicePhase = "pending"
	MicroServicePhaseStopped MicroServicePhase = "stopped"
	MicroServicePhaseUnknown MicroServicePhase = "unknown"
)

// MicroServiceType ...
type MicroServiceType string

const (
	MicroServiceTypeStateful  MicroServiceType = "stateful"
	MicroServiceTypeStateless MicroServiceType = "stateless"
)

// MicroService ...
type MicroService struct {
	AppName           string             `json:"appName"`
	Project           string             `json:"projectName"`
	Name              string             `json:"name"`
	GPUSpec           *GPUSpec           `json:"gpuSpec,omitempty"`
	Status            MicroServiceStatus `json:"status,omitempty"`
	Pods              []MicroServicePod  `json:"pods"`
	Type              MicroServiceType   `json:"type"`
	ResourceSpec      string             `json:"resourceSpec"`
	InstanceNumber    int32              `json:"instanceNumber"`
	MicroServicePorts []MicroServicePort `json:"ports"`
	Containers        []Container        `json:"containers"`
	CTime             time.Time          `json:"creationTime"`
}

type GPUSpec struct {
	Type   string `json:"type"`
	Number int32  `json:"number"`
}

type MicroServiceStatus struct {
	Phase       MicroServicePhase `json:"phase,omitempty"`
	Desire      int32             `json:"desire,omitempty"`
	Current     int32             `json:"current,omitempty"`
	Updated     int32             `json:"updated,omitempty"`
	Available   int32             `json:"available,omitempty"`
	Unavailable int32             `json:"unavailable,omitempty"`
}

type PodPhase string

const (
	PodPhaseRunning   PodPhase = "running"
	PodPhasePending   PodPhase = "pending"
	PodPhaseSucceeded PodPhase = "succeeded"
	PodPhaseFailed    PodPhase = "failed"
	PodPhaseUnknown   PodPhase = "unknown"
)

type MicroServicePod struct {
	Name              string            `json:"name,omitempty"`
	Status            PodStatus         `json:"status,omitempty"`
	ContainerStatuses []ContainerStatus `json:"containerStatuses"`
}
type PodStatus struct {
	Phase   PodPhase `json:"phase,omitempty"`
	Reason  string   `json:"reason,omitempty"`
	Message string   `json:"message,omitempty"`
}
type Pod struct {
	MicroServicePod
	AppName          string      `json:"appName,omitempty"`
	MicroServiceName string      `json:"microserviceName,omitempty"`
	ProjectName      string      `json:"projectName,omitempty"`
	Containers       []Container `json:"containers,omitempty"`
}

type ContainerState string

const (
	ContainerStateRunning    ContainerState = "running"
	ContainerStateWaiting    ContainerState = "waiting"
	ContainerStateTerminated ContainerState = "terminated"
	ContainerStateUnknown    ContainerState = "unknown"
)

type ContainerStatus struct {
	Name    string         `json:"name,omitempty"`
	State   ContainerState `json:"state,omitempty"`
	Reason  string         `json:"reason,omitempty"`
	Message string         `json:"message,omitempty"`
}

// MicroServiceUpgradeArgs ...
type MicroServiceUpgradeArgs struct {
	ResourceSpec string      `json:"resourceSpec"`
	Containers   []Container `json:"containers"`
}

// MicroServiceUpdatePortsArgs ...
type MicroServiceUpdatePortsArgs struct {
	MicroServicePorts []MicroServicePort `json:"ports"`
}

// Container ...
type Container struct {
	Name                  *string                `json:"name,omitempty"`
	Image                 string                 `json:"image"`
	WorkingDir            string                 `json:"workingDir"`
	Command               []string               `json:"command"`
	Args                  []string               `json:"args"`
	ContainerVolumeMounts []ContainerVolumeMount `json:"volumeMounts"`
	ContainerConfigs      []ContainerConfig      `json:"configs"`
	ContainerEnvs         []ContainerEnv         `json:"envs"`
}

// ContainerVolumeMount ...
type ContainerVolumeMount struct {
	VolumeName string `json:"volumeName"`
	ReadOnly   *bool  `json:"readOnly"`
	// TODO: Validation Path
	MountPath string `json:"mountPath"`
}

// ContainerConfig ...
type ContainerConfig struct {
	ConfigMapName string `json:"configMapName"`
	// TODO: Validation Path
	MountPath string `json:"mountPath"`
}

// ContainerEnvType ...
type ContainerEnvType string

const (
	ContainerEnvTypeConfigMap ContainerEnvType = "configMap"
)

// ContainerEnv ...
type ContainerEnv struct {
	Type  ContainerEnvType `json:"type,omitempty"`
	Name  string           `json:"name"`
	Value string           `json:"value"`
}

// ProtocolType ...
type ProtocolType string

const (
	ProtocolTypeTCP ProtocolType = "TCP"
	ProtocolTypeUDP ProtocolType = "UDP"
)

// MicroServicePort ...
type MicroServicePort struct {
	Port        int32        `json:"port"`
	Protocol    ProtocolType `json:"protocol"`
	ServicePort int32        `json:"servicePort,omitempty"`
}

// MicroserviceRevision ...
type MicroserviceRevision struct {
	Revision          int64       `json:"revision"`
	ResourceSpec      string      `json:"resourceSpec"`
	Containers        []Container `json:"containers"`
	CreationTimestamp time.Time   `json:"creationTimestamp"`
}
