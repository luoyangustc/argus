# argus

## 目录说明

```bash
docs        # 综合设计文档
hack        # 开发环境脚本
docker      # 镜像构建
```

## 本地开发环境

首先安装依赖：

- golang 1.10.x
- mongo 3.x
- etcd v3
- nsq

然后运行 mongo

运行 etcd，nsq `make serving-run-base`

运行服务组，如 `make serving-run-eval` 或者 `make serving-run-tuso`

服务启动之后，修改代码或者配置文件，对应服务会自动重启，如果有问题 @wangkechun

常用命令：

./hack/run/bin/goreman -p 8555 run status 可以查看服务运行状态

./hack/run/bin/goreman -p 8555 run restart \<appName\> 可以重启服务

goreman 端口默认 8555，可以省略，对应服务组的端口在 Makefile 可以找到

## 工作流程规范

提交代码之前准备，需要运行 `make` 检查代码编译、代码风格、运行单元测试

golangci-lint 需要自行下载到环境变量，然后配置编辑器集成 https://github.com/golangci/golangci-lint#editor-integration

建立 Pull requests 之前，请先运行 `git fetch -f git@github.com:qbox/argus.git dev:qiniu-dev && git rebase qiniu-dev` rebase 代码，然后重新 `make`

## 依赖管理

统一使用 govendor 来管理 https://cf.qiniu.io/pages/viewpage.action?pageId=22708258

## Production

![Coverage](https://aone.qiniu.io/api/coverage/badge.svg?token=A59B8029-E9A1-4FF3-B72B-3657CEAC64D4&repo=qbox/ava)
