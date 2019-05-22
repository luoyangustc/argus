# RAS tool

目前，AI 私有化项目的交付的形式为 ansible playbook + docker，交付实施人员需要掌握相关细节配置

ras-tool 的目的是封装底层的技术细节，让实施人员专注于私有化项目部署的业务

## 功能

ras-tool 将提供以下功能：

* 私有化环境的初始化，包括：依赖软件下载、安装，镜像/模型下载
* 私有化服务编排
* 私有化服务安装、部署、运行
* 私有化服务版本管理
* 私有化服务自检

## 安装

TODO

## 命令

### 初始化环境

```bash
ras-tool init

ras-tool init <project_name>
```

### 运行私有化服务

```bash
ras-tool run <project_name>
```

### 显示私有化服务信息

```bash
ras-tool info <project_name>

xxx_project
    facex-detect:
        image: hub2.qiniu.com/1381102897/ava-eval-face.face-det.tron:201804132200
        model: ava-model ava-facex-detect/tron-refinenet/201803231039.tar
        args:
            xxx
    serving-gate:
        image: hub2.qiniu.com/1381102897/ava-serving-gate:201802051900
    ...
```

### 显示私有化服务历史版本

```bash
ras-tool history <project_name>

# git log history
```

### 切换私有化服务版本

```bash
ras-tool switch <project_name> <version>
```

### 检验私有化服务运行状态

```bash
ras-tool check <project_name>
```