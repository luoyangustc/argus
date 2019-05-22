# Ras-tool 项目配置格式

Ras-tool 对应的 ava-enterprise 目录结构

```bash
templates                       // ras-tool 所需要的模板
├── project.yml                 // 项目主体工程模板
├── inventories                 // 机器列表/变量模板
│   ├── group_vars
│   │   ├── all.yml
│   ├── host.yml
projects
├── standards                    // 固化下来的服务模块
│   ├── sanjian.yml
├── 20180511-jz-terror-face.yml  // 正式私有化部署的项目配置文件
├── xxx.yml
playbook
├── [project.yml]               // 自动生成
├── [inventories]               // 自动生成
├── roles                       // playbook 中只负责维护 role
│   ├── argus_gate
│   ├── argus_video
│   ├── serving_eval
│   ├── serving_gate
│   ├── ...
```

project 私有化部署整体配置文件格式

```yaml
hosts:                       // host 配置
  master:
    - 100.100.62.217
  worker:
    - 100.100.62.217
    - 100.100.62.218
    - ...

services:                       // 服务列表

    master:
        mongo:
        argus_gate:
            image: "hub2.qiniu.com/1381102897/ava-argus-util:201805101531_fix"

    worker:
        serving_gate:
            image: "hub2.qiniu.com/1381102897/ava-serving-gate:201802051900"
        serving_eval.facex-detect:
            image: "hub2.qiniu.com/1381102897/ava-eval-face.face-det.tron:201804132200"
            models:
            - path: "ava-facex-detect/tron-refinenet/201803231039.tar"
            instance: 2
            use_device: GPU
            gpu_index: [0,1]
        serving_eval.facex-feature-v2:
            image: "hub2.qiniu.com/1381102897/ava-eval-caffe-facex-feature-v2:201712202010"
            models:
            - path: "ava-facex-feature/caffe-facex-feature-v2/201709212209.tar"
            instance: 1
            use_device: GPU
            gpu_index: [2]
            args:
            batch_size:       1
            image_width:      96
            custom_values:
                image_height:   112
                input_scale:    0.0078125
                workspace:      /tmp/eval/
        serving_eval.politician:
            image: "hub2.qiniu.com/1381102897/ava-eval-other-facex-search:201803081209"
            models:
            - path: "ava-politician/other-face-search/201803081512.features.line"
            - path: "ava-politician/other-face-search/201803021746.labels.line"
```