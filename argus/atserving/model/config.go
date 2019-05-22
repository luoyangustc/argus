package model

import (
	"errors"
	"fmt"
	"regexp"
	"time"
)

// ETCD中KEY的组织
const (
	KeyPrefix = "/ava/serving"

	KeySTS         = KeyPrefix + "/sts/hosts"
	KeyNSQConsumer = KeyPrefix + "/nsq/consumer"
	KeyNSQProducer = KeyPrefix + "/nsq/producer"
	KeyModelT      = KeyPrefix + "/model/%s/%s"

	KeyLogPush           = KeyPrefix + "/logpush"
	KeyWorkerPrefix      = KeyPrefix + "/worker"
	KeyWorkerDefault     = KeyWorkerPrefix + "/default"
	keyWorkerAppT        = KeyWorkerPrefix + "/app/%s/default"
	keyWorkerAppReleaseT = KeyWorkerPrefix + "/app/%s/release/%s"

	KeyAppPrefix          = KeyPrefix + "/app"
	KeyAppDefaultPrefix   = KeyAppPrefix + "/default"
	KeyAppMetadataDefault = KeyAppDefaultPrefix + "/metadata"
	KeyAppMetadataPrefix  = KeyAppPrefix + "/metadata"
	keyAppMetadataT       = KeyAppMetadataPrefix + "/%s"
	KeyAppReleasePrefix   = KeyAppPrefix + "/release"
	keyAppReleaseT        = KeyAppReleasePrefix + "/%s/%s"
)

// 生成ETCD中的KEY
var (
	KeyWorkerApp        = func(app string) string { return fmt.Sprintf(keyWorkerAppT, app) }
	KeyWorkerAppRelease = func(app, version string) string { return fmt.Sprintf(keyWorkerAppReleaseT, app, version) }

	KeyAppMetadata = func(app string) string { return fmt.Sprintf(keyAppMetadataT, app) }
	KeyAppRelease  = func(app, version string) string { return fmt.Sprintf(keyAppReleaseT, app, version) }

	KeyModel = func(problemType, framework string) string { return fmt.Sprintf(KeyModelT, problemType, framework) }
)

// ConfigEtcd 访问ETCD的配置文件
type ConfigEtcd struct {
	DialTimeoutMs int64 `json:"dail_timeout_ms"`
}

// ConfigSTSHost STS节点配置
type ConfigSTSHost struct {
	Key  string `json:"key"`
	Host string `json:"host"`
}

// ConfigConsumer 队列消费者配置
type ConfigConsumer struct {
	Addresses   []string `json:"addresses"`
	Topic       *string  `json:"topic,omitempty"`
	Channel     *string  `json:"channel,omitempty"`
	MaxInFlight *int     `json:"max_in_flight,omitempty"`
}

//----------------------------------------------------------------------------//

// ConfigWorker 任务执行者配置
type ConfigWorker struct {
	Timeout       time.Duration `json:"timeout"`
	MaxConcurrent int64         `json:"max_concurrent"`
	Deplay4Batch  int64         `json:"delay4batch"`
	Kodo          struct {
		RsHost string `json:"rs_host"`
	} `json:"kodo"`
}

//----------------------------------------------------------------------------//

// ConfigModel 模型相关配置
type ConfigModel struct {
	Image  string `json:"image"`
	Flavor string `json:"flavor"`
	UseCPU bool   `json:"use_cpu"`
}

//----------------------------------------------------------------------------//

// Owner 用户信息
type Owner struct {
	UID uint32 `json:"uid"`
	AK  string `json:"ak"`
	SK  string `json:"sk"`
}

// ConfigAppMetadata 应用的基本信息
type ConfigAppMetadata struct {
	BatchSize int64 `json:"batch_size"`
	Public    bool  `json:"public"`
	Owner     Owner `json:"owner"`

	UserWhiteList []uint32 `json:"user_whitelist"`
}

// ConfigAppRelease 应用各Release的配置信息
type ConfigAppRelease struct {
	TarFile string `json:"tar_file"`

	ImageWidth int    `json:"image_width"`
	BatchSize  *int64 `json:"batch_size,omitempty"`

	CustomFiles  map[string]string      `json:"custom_files,omitempty"`
	CustomValues map[string]interface{} `json:"custom_values,omitempty"`

	Phase string `json:"phase"` //staging|production

	ModelParams interface{} `json:"model_params,omitempty"`
}

//----------------------------------------------------------------------------//

var (
	// ErrBadConfigKey error
	ErrBadConfigKey = errors.New("bad config key")
)

// ConfigKeyWorker worker的唯一表示
type ConfigKeyWorker struct {
	App     *string
	Version *string
}

var (
	_ReWorkerApp        = regexp.MustCompile(KeyWorkerPrefix + "/app/([^/]+)/default")
	_ReWorkerAppRelease = regexp.MustCompile(KeyWorkerPrefix + "/app/([^/]+)/release/([^/]+)")
)

// Parse parse
func (key *ConfigKeyWorker) Parse(bs []byte) error {
	var s = string(bs)
	if s == KeyWorkerDefault {
		key.App, key.Version = nil, nil
		return nil
	}
	if rest := _ReWorkerApp.FindStringSubmatch(s); rest != nil {
		key.App = &rest[1]
		key.Version = nil
		return nil
	}
	if rest := _ReWorkerAppRelease.FindStringSubmatch(s); rest != nil {
		key.App = &rest[1]
		key.Version = &rest[2]
		return nil
	}
	fmt.Println(s)
	return ErrBadConfigKey
}

// ConfigKeyAppMetadata AppMetadataKey
type ConfigKeyAppMetadata struct {
	App string
}

var (
	_ReAppMetadata = regexp.MustCompile(KeyAppMetadataPrefix + "/([^/]+)")
)

// Parse function parse
func (key *ConfigKeyAppMetadata) Parse(bs []byte) error {
	var s = string(bs)
	if rest := _ReAppMetadata.FindStringSubmatch(s); rest != nil {
		key.App = rest[1]
		return nil
	}
	fmt.Println(s)
	return ErrBadConfigKey
}

// ConfigKeyAppRelease AppReleaseKey
type ConfigKeyAppRelease struct {
	App     string
	Version string
}

var (
	_ReAppRelease = regexp.MustCompile(KeyAppReleasePrefix + "/([^/]+)/([^/]+)")
)

// Parse function parse
func (key *ConfigKeyAppRelease) Parse(bs []byte) error {
	var s = string(bs)
	if rest := _ReAppRelease.FindStringSubmatch(s); rest != nil {
		key.App = rest[1]
		key.Version = rest[2]
		return nil
	}
	fmt.Println(s)
	return ErrBadConfigKey
}
