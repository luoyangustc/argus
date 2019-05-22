# Argus Service

## 编译

APP 编译通过 main 函数模板 + APP 编译信息确定，其中，APP 编译信息格式为：

```json
{
    "app"       : "app_name",
    "version"   : "app_release_or_other_uniq_string",
    "scenario"  : "image_sync",
    "services": {
        "pulp"  : {},       // default package: "qiniu.com/argus/service/service/image/pulp/image_sync"
        "terror": {
            "package"   : "qiniu.com/3rd/service/package"
        }
    }
}
```

根据模板和编译信息生成 APP main.go 之后，编译出 APP binary

## APP 导出信息规范

APP binary 需支持导出各类元信息，用于和交付、部署环境进行集成，元信息包括：

* 监控
* 路由

APP 导出元信息规范

APP 运行的命令为：

```bash
<APP> -f app.conf [-mock]
```

约定 APP 会在启动初始化完成之后，将 APP info 信息写入 `<workspace>/integration` 文件夹，如果设置了 `-mock` flag，则表示各中间件在进行 mock 初始化。

输出文件夹结构：

```bash
<workdir>/integration
    monitor
        monitor.conf.json
        dashboards
            dashboard-name-1.data.json
            ...
    router
        router.conf.json
        api.json
```

输出文件夹结构：

```bash
<workdir>/integration
    monitor
        monitor.conf.json
        dashboards
            dashboard-name-1.data.json
            ...
    router
        router.conf.json
        api.json
```

其中：

`monitor.conf.json` 格式

```json
{
    "scrap_path": "/metrics"
}
```

`dashboards/xxx.json` 中的格式为 Grafana dashboard 格式

`router.conf.json` 格式

```json
{
    "srv-name-1": {
        "prefix": "",
        "redirect": {
            "path1": "redirect_path1",
            "path2": "redirect_path2",
        }
    },
    "srv-name-2": {
        ...
    }
}
```

`api.json` 格式

```json
{
    "srv-name-1": [
        {
            "prefix": "",
            "method": "POST",
            "path": "v1/pulp",
            "desc": "pulp API markdown doc"
        },{
            ...
        }
    ],
    "srv-name-2": [
        {
            "prefix": "",
            "method": "POST",
            "path": "v1/terror",
            "desc": "terror API markdown doc"
        }, {
            ...
        }
    ]
}
```

## 集成

集成步骤：

### 服务部分

1. 根据 APP 编译信息，生成 main 函数，编译出 APP binary
2. 执行 APP info 命令，生成静态默认配置数据 `integration.default`，包括默认的支撑数据（监控 Dashboard / API 文档 / ...）
3. 将 APP binary 编入 image
4. 将 `integration.default` 上传至 bucket

### 私有化项目工程部分

1. 在指定的 `<scenario>/<app>` 中更新 APP 版本，以及 APP conf （port / router / service conf / ...）

（每个场景决定了有哪些依赖服务，以及服务的依赖关系）

### 私有化部署部分

1. 下载各服务镜像、模型，生成配置
2. 启动基础服务（不依赖 APP）
3. 启动 APP，生成 APP 在私有化环境下的动态配置数据 `integration`，生成各支撑服务的配置和数据、以及交付数据（例如：API 文档）
4. 启动依赖于 APP 的服务