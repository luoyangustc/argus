package util

import (
	"qiniu.com/argus/test/configs"
)

func GetOp(server string, testtype string) string {
	return configs.StubConfigs.Servers.Type[testtype][server].Op
}
