package proto

import (
	"context"

	"qiniu.com/argus/tuso/io"
)

type HubInfo struct{}
type HubMetaInfo struct {
	HubName          string
	FeatureVersion   string
	FeatureFileIndex int
}

type HubMgr interface {
	HubInfo(context.Context, string) (HubInfo, error)
	HubMetaInfo(context.Context, string) (HubMetaInfo, error)
}

////////////////////////////////////////////////////////////////////////////////

type LogInfo struct{}
type LogMgr interface {
}

////////////////////////////////////////////////////////////////////////////////

type ImageInfo struct{}

type MetaMgr interface {
	ImageInfo(context.Context,
		string, // hub
		string, // key
	) (ImageInfo, error)
}

////////////////////////////////////////////////////////////////////////////////

type FeatureFileMgr interface {
	Open(context.Context,
		string, // hub
		string, // feature version
	) (io.Blocks, error)
}
