# ai-sdk

## 参考资料：

[【07-02-00-01 方案】Image SDK 的交付集成形式](https://cf.qiniu.io/pages/viewpage.action?pageId=3549745)

[构建高性能深度学习推理服务](https://cf.qiniu.io/pages/viewpage.action?pageId=2164982)

## 入门指南

[构建高性能深度学习推理服务-AI-SDK 入门指南](https://cf.qiniu.io/pages/viewpage.action?pageId=2164982#id-%E6%9E%84%E5%BB%BA%E9%AB%98%E6%80%A7%E8%83%BD%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E6%8E%A8%E7%90%86%E6%9C%8D%E5%8A%A1-AI-SDK%E5%85%A5%E9%97%A8%E6%8C%87%E5%8D%97)

## flavor 机制

[flavor 机制](python/aisdk/common/flavor.py)

## GPU_INDEX 机制

支持通过环境变量选择使用的 GPU
[GPU_INDEX 机制](python/aisdk/common/ai_sdk.py)

## vscode cpp 自动补全配置指南

```bash
cd ataraxia/ai-sdk
brew install cmake ninja clang-format llvm
brew install protobuf # 确认版本是 3.6.1
brew install opencv glog zmq boost openblas
cd cpp && mkdir -p build && cd build
cmake -DCMAKE_BUILD_TYPE=DEBUG ..
会生成 cpp/build/compile_commands.json
然后vscode安装c++插件，以aisdk为根目录重新打开项目，然后点开一个c++文件会提示配置 c_cpp_properties.json，参考下面的设置即可
```

```json
{
  "configurations": [
    {
      "name": "Mac",
      "includePath": [],
      "defines": [],
      "macFrameworkPath": [],
      "compileCommands": "${workspaceFolder}/cpp/build/compile_commands.json",
      "compilerPath": "/usr/bin/clang++"
    }
  ],
  "version": 4
}
```

然后打开任一 c++ 代码文件，应该代码跳转和自动补全都能正常使用
此时可以 make -j 4 来在 mac 下面编译项目 （在 mac 下面部分代码只编译不链接）
vscode 请打开 `"editor.formatOnSave": true` 配置
clion 可以参考 https://www.jetbrains.com/help/clion/compilation-database.html 自行配置

## 转存的第三方依赖管理

统一放在 avatest@qiniu.com bucket:devtools
域名为 http://devtools.dl.atlab.ai

### 来自 github release

aisdk 依赖的 github release 里面的文件统一前缀 aisdk/github_release/
比如 cmake 的下载地址原始地址为： https://github.com/Kitware/CMake/releases/download/v3.13.2/cmake-3.13.2-Linux-x86_64.sh
加速下载地址为： http://devtools.dl.atlab.ai/aisdk/github_release/Kitware/CMake/releases/download/v3.13.2/cmake-3.13.2-Linux-x86_64.sh

### 自行处理过的文件

统一前缀 aisdk/other/<来源>

#### mxnet 重新打包

打包命令，把 submodule 打包进去了：

```bash
git clone -b v1.3.1 --depth 1 --recursive https://github.com/apache/incubator-mxnet mxnet && \
git submodule init && git submodule update && make clean
```

下载地址：
http://devtools.dl.atlab.ai/aisdk/other/incubator-mxnet-1.3.1.tar.gz

### 其它文件

统一前缀 aisdk/other/<来源>

如 http://devtools.dl.atlab.ai/aisdk/other/from_tensorrt/opencv-3.4.1.tar
来自 https://github.com/qbox/ataraxia/blob/abdf2cfb5ba3f5615f5178e1c2c8a6ad0b46941f/inference/face/face-det/tensorrt/docker/Dockerfile#L29
