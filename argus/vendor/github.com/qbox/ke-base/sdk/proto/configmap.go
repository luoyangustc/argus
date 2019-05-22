package proto

import "time"

// ConfigMap ...
type ConfigMap struct {
	Name  string            `json:"name"`
	Data  map[string]string `json:"data"`
	CTime time.Time         `json:"creationTime"`
}