package util

import (
	"qiniu.com/argus/test/configs"
)

func GetPath(server string, testtype string, pathtype string) string {
	if pathtype == "Path" || pathtype == "path" {
		return configs.StubConfigs.Servers.Type[testtype][server].Version + configs.StubConfigs.Servers.Type[testtype][server].Path
	} else if pathtype == "EvalPath" || pathtype == "evalpath" {
		return configs.StubConfigs.Servers.Type[testtype][server].Version + configs.StubConfigs.Servers.Type[testtype][server].EvalPath
	} else {
		return ""
	}
}
