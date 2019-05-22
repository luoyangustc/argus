package client

import (
	"time"
)

type MetaData struct {
	TarURI     string                 `json:"tar_uri"`
	OtherFiles map[string]string      `json:"other_files"`
	Configs    map[string]interface{} `json:"configs"`
}

type Release struct {
	App        string    `bson:"app"`
	Version    string    `bson:"version"`
	CreateTime time.Time `bson:"create_time"`
	MetaData   MetaData  `bson:"metadata"`
	Desc       string    `bson:"desc"`
}
