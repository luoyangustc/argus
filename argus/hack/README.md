<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->
**Table of Contents**  *generated with [DocToc](https://github.com/thlorenz/doctoc)*

- [AVA 开发环境 - AtNet/AtFlow](#ava-%E5%BC%80%E5%8F%91%E7%8E%AF%E5%A2%83---atnetatflow)
  - [k8s 环境](#k8s-%E7%8E%AF%E5%A2%83)
    - [minikube](#minikube)
    - [docker for mac](#docker-for-mac)
    - [kubectl](#kubectl)
  - [准备工作 - 本地 db 及 mock 七牛鉴权服务](#%E5%87%86%E5%A4%87%E5%B7%A5%E4%BD%9C---%E6%9C%AC%E5%9C%B0-db-%E5%8F%8A-mock-%E4%B8%83%E7%89%9B%E9%89%B4%E6%9D%83%E6%9C%8D%E5%8A%A1)
  - [build app & run app](#build-app-&-run-app)
  - [集成 cases](#%E9%9B%86%E6%88%90-cases)
    - [global configuration](#global-configuration)
    - [基本流程](#%E5%9F%BA%E6%9C%AC%E6%B5%81%E7%A8%8B)
    - [其他 cases](#%E5%85%B6%E4%BB%96-cases)
- [AVA 开发环境 - Serving 本地环境](#ava-%E5%BC%80%E5%8F%91%E7%8E%AF%E5%A2%83---serving-%E6%9C%AC%E5%9C%B0%E7%8E%AF%E5%A2%83)
    - [测试方法：](#%E6%B5%8B%E8%AF%95%E6%96%B9%E6%B3%95%EF%BC%9A)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->

# AVA 开发环境 - AtNet/AtFlow
支持 AtNet/AtFlow API 及 worker 在 k8s 环境运行

## k8s 环境
minikube or docker 自带 (prefer)
注：docker 版本需要 17.05 以上

### minikube
https://kubernetes.io/docs/getting-started-guides/minikube/

使用 xhyve 启动 minikube，无需 virtualbox；首次启动配置资源：内存至少 4G，建议 6G 以上; 磁盘建议 40g 以上。
```bash
$ minikube start --memory 4096 --disk-size 40g --vm-driver=xhyve
```

在 minikube 环境下执行命令（比如构建镜像）需要：
`eval $(minikube docker-env)`

### docker for mac
新版自带 k8s: https://blog.docker.com/2018/01/docker-mac-kubernetes/

dashboard 需要手动安装: https://github.com/kubernetes/dashboard

### kubectl
kubectl 自动补全和 exec 宽度:
```
# k8s, add to ~/.bashrc
source <(kubectl completion bash)
kubessh() {
  if [ "$1" == "" ]; then
    echo "Usage: kubessh <pod>"
    exit 1
  fi
  cmd=$2
  if [ "$cmd" == "" ]; then
    cmd=bash
  fi
  COLUMNS=`tput cols`
  LINES=`tput lines`
  TERM=xterm
  kubectl exec -i -t $1 env COLUMNS=$COLUMNS LINES=$LINES TERM=$TERM $cmd
}
```

## 准备工作 - 本地 db 及 mock 七牛鉴权服务

* 大部分 app 依赖 mongo，在宿主机或容器运行的 mongo 需要开放 `:27017` 的访问权限。
* 账号相关服务。
  * https://github.com/qbox/api.qiniu.com
  * https://github.com/qbox/biz

```bash
[qbox]$ git clone git@github.com:qbox/base.git
[qbox]$ git clone git@github.com:qbox/api.qiniu.com.git
[qbox]$ git clone git@github.com:qbox/biz.git

[ava]$ hack/atnet/init-acc.sh
# 手动替换 hack/run/atnet/qboxaccess.conf 和 hack/run/mockacc.conf 中 key/secret 为线上 avatest@qiniu.com (uid: 1381102889) 的 AK/SK
# 将 hack/run/atnet/qboxone.conf 中的 mongodb 地址，改为本地测试的地址 (optional)
# 将 hack/run/atnet/qboxone.conf 中的 `access_proxy.appd_access_host` 改为 `http://localhost:13211` (optional)
[ava]$ hack/atnet/service.sh startacc
# [ava]$ hack/atnet/service.sh stopacc
```
* 注：因为当前的 case 尚未使用本地的 kodo/dora 测试，都依赖线上 acc ，所以有手动替换 avatest@qiniu.com 的步骤

## build app & run app

```bash
# 拉取基础及第三方镜像 (reg-xs.qiniu.io 帐号请在 xs713 上获取)
[ava/hack/atnet]$ ./install-image-k8s.sh atnet-apigate:20170626-0522516
[ava/hack/atnet]$ ./install-image-k8s.sh third-party

# 构建训练镜像
[ava/hack/atnet]$ ./build.sh ava-public-base
[ava/hack/atnet]$ ./build.sh ava-public

编译所有比较慢，建议只编 mxnet cpu 版本(https://github.com/qbox/ava/docker/app/ava-public/ava-mxnet-py27-cpu.Dockerfile)，需要在 build 脚本里手动过滤一下。

# 构建其他服务镜像
[ava/hack/atnet]$ ./build.sh <app>

# 注：会执行 ./hack/app/update-ip.sh <app>
# 更新 en0 网卡地址到 app 的 mongodb, 本地 qiniu 服务的 host 配置
[ava/hack/atnet]$ ./service.sh reborn <app>

# 查看状态
[ava/hack/atnet]$ ./service.sh status <app>

# 清除 job 和服务
[ava/hack/atnet]$ ./service.sh clean && ./service.sh sweep
# 清除 job 并重新部署服务
[ava/hack/atnet]$ ./service.sh reborn

# 查看 apiserver 地址
[ava/hack/atnet]$ ./service.sh api

# 调试：用 sleep 重启/回滚
[ava/hack/atnet]$ ./patch-sleep.sh sampleset-controller
[ava/hack/atnet]$ kubectl rollout undo deploy/sampleset-controller
```

## 集成 cases

### 用户及配置

```
# 运行集成 cases 之前，先创建用户
[ava/hack/atnet]$ ./create-user.sh
```

### 基本流程

```
# 创建数据集并启动下载任务
[ava/hack/atnet]$ ./create-spec.sh ./mock/datasetspec/classification-terror.json

# 创建格式化数据集并启动构建任务
[ava/hack/atnet]$ ./create-spec.sh ./mock/samplesetspec/classification-recordio.json

# 创建并启动训练
[ava/hack/atnet]$ ./create-spec.sh ./mock/trainingspec/classification-mxnet.json

# 启动训练实例(example)，登录容器后
[root@host]$ /workspace/examples/trainings/mxnet/start.sh
```

* 注：如果因磁盘空间不足失败
  * `./service.sh sweep` 清理 job
  * 登陆 `minikube ssh` 后 `sudo rm -rf /mnt/sda1/ava-hack` 。

### 其他 cases
建议使用 postman import 后调用 api
```
https://github.com/qbox/ava/hack/atnet/atnet-api.postman_collection.json
```


# AVA 开发环境 - Serving 本地环境

- 安装 nsq、etcd
- cd $QBOXROOT/ava && source env-ci.sh
- make serving-run-etcd 跑起来etcd的组件（如果etcd已经作为系统服务跑起来了，跳过这一步）
- make serving-run-nsq 跑起来nsq的组件，nsqadmin地址： http://localhost:4171/
- make serving-init-etcd 初始化etcd配置
- make serving-run-other 跑起来serving组件，goreman run status 查看状态
- make watch 启用自动重新编译重启组件，可在 hack/serving/modd/modd.conf 添加规则

### 测试方法：

```
curl -v "http://127.0.0.1:9101/v1/eval/hello_eval" -X "POST" -H "Content-Type: application/json" -d '{"data":{"uri":"http://www.qiniu.com"}}'

curl -v "http://127.0.0.1:9101/v1/groupeval/hello_groupeval" -X "POST" -H "Content-Type: application/json" -d '{"data":[{"uri":"http://www.qiniu.com"}]}'

curl -v "http://127.0.0.1:9101/v1/eval/batch/hello_eval" -X "POST" -H "Content-Type: application/json" -d '[{"data":{"uri":"http://www.qiniu.com"}},{"data":{"uri":"http://www.qiniu.com"}}]'
```
