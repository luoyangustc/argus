package proto

// RegionConfig ...
type RegionConfig struct {
	ResourceSpecs []ResourceSpec         `json:"resourceSpecs"`
	Feature       FeatureConfig          `json:"feature"`
	Alert         map[string]interface{} `json:"alert"`
}

// FeatureConfig ...
type FeatureConfig struct {
	TLBEnabled      *bool `json:"tlbEnabled"`
	AppStoreEnabled *bool `json:"appStoreEnabled"`
}

// ResourceSpec ...
type ResourceSpec struct {
	Name string `json:"name"`
	// CPU 单位: 核数
	CPU int32 `json:"cpu"`
	// Memory 单位: MB
	Memory int32  `json:"memory"`
	Desc   string `json:"desc"`
}
