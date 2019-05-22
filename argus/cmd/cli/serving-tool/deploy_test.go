package main

import (
	"testing"

	"github.com/stretchr/testify/assert"
	yaml "gopkg.in/yaml.v2"
)

func TestFoo(t *testing.T) {
	{
		var v interface{} = new(interface{})
		assert.Nil(t, *v.(*interface{}))
	}
	{
		assert.True(t, isNil(nil))
		var v interface{} = new(interface{})
		assert.True(t, isNil(v))
		assert.True(t, isNil(cleanupMapValue(v)))
	}
}

func TestDeploy(t *testing.T) {

	var appyaml = `
---

release:
  flavor: C4M4 # 运行实例的机器配置，不同的配置单实例价格不一样，使用 qdoractl flavor 命令获取可用的配置列表
  env:  # app 启动的时候附加的环境变量，global 代表所有 region 都有的环境变量、z0 代表 z0 region 独有的环境变量
    global:
      - key: SERVING_GATE_HOST
        value: http://ava-serving-gate.xs.cg.dora-internal.qiniu.io:5001

  log_file_paths: # OPTIONAL 用户日志路径，会采集该路径下的用户日志，系统也会对该目录下的已采集日志进行自动回收。
    - "/workspace/serving/run/auditlog/ARGUS_GATE"
`
	var data appConfig
	err := yaml.Unmarshal([]byte(appyaml), &data)
	assert.NoError(t, err)
	assert.Nil(t, data.Config.Meta)
	assert.Nil(t, data.Config.Release)

}
