package config

import (
	"qbox.us/cc/config"

	"qiniu.com/argus/argus/facec/dbbase"
)

// Service algorithm API configures
type Service struct {
	Name          string `json:"name"`
	URL           string `json:"url"`
	MaxConcurrent int    `json:"max_concurrent"` // the instance count
	MaxImageCount int    `json:"max_image_count"`
}

// API configures the algorithm API
type API struct {
	AK       string    `json:"ak"`
	SK       string    `json:"sk"`
	Services []Service `json:"services"`
}

// ClusterWorker configure the background job action
type ClusterWorker struct {
	Period          int     `json:"period"`
	UserCount       int     `json:"user_count"`
	FacexFeatureAPI Service `json:"facex_feature_api"`
	FacexClusterAPI Service `json:"facex_cluster_api"`
}

// Argus configures
type Argus struct {
	Version string    `json:"verson"`
	DB      dbbase.DB `json:"db"`
	API     API       `json:"api"`
}

// GlobalArgus readonly argus's configure
var GlobalArgus Argus

// Load the argus configures
func Load() (*Argus, error) {
	config.Init("f", "argus-gate", "argus-gate.conf")
	err := config.Load(&GlobalArgus)
	if err == nil {
		return &GlobalArgus, nil
	}
	return nil, err
}
