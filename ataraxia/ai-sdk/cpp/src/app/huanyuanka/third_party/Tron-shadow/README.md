# Atlab SDK TEAM Inference LIB


### 介绍

项目包含内容：

- Tron-shadow/3rdparty            依赖库
- Tron-shadow/algorithm/ssd       算法示例/ssd通用检测
- Tron-shadow/algorithm/refinedet 算法示例/refinedet通用检测
- Tron-shadow/bj_run              bj_run五合一
- Tron-shadow/Landmark3D          人脸3d landmark
- Tron-shadow/Vsepa               Vsepa行人检测网络
- Tron-shadow/mixup               暴恐三模型
### 依赖配置

1. [下载第三方库](#第三方库)
2. [安装OpenCV-3.1.4-CUDA](#OpenCV-3.1.4-CUDA)


#### 下载第三方库

1. 请根据设备环境的CUDA以及CUDNN配置下载相应版本的第三方库，下载位置为项目主目录中

   - [3rdparty-tensorrt4.0.1.6-cuda9.0-cudnn7.1](http://pbv7wun2s.bkt.clouddn.com/3rdparty-tensorrt4.0.1.6-cuda9.0-cudnn7.1-radpidjson.zip)

   - [3rdparty-tensorrt4.0.1.6-cuda8.0-cudnn7.1](http://pbv7wun2s.bkt.clouddn.com/3rdparty-tensorrt4.0.1.6-cuda8.0-cudnn7.1-radpidjson.zip)


2. 解压文件夹

  ```Shell
  unzip 3rdparty-*.zip
  ```


#### 安装OpenCV-3.1.4-CUDA

1. 任意路径下载[opencv3.4.1](http://pbv7wun2s.bkt.clouddn.com/opencv-3.4.1.tar)

2. 在opencv3.4.1下载位置依次执行以下命令 

  ```Shell
  tar -xvf opencv-3.4.1.tar && rm opencv-3.4.1.tar
  cd opencv-3.4.1
  mkdir build 
  cd build
  cmake -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=/usr/local -D WITH_CUDA=ON -D ENABLE_FAST_MATH=ON -D CUDA_FAST_MATH=ON -D WITH_CUBLAS=1 -D WITH_NVCUVID=on -D CUDA_GENERATION=Auto ..
  make -j*
  make install
  ```


# Atlab SDK TEAM Inference LIB


### 介绍

项目包含内容：

- Tron-shadow/3rdparty            依赖库
- Tron-shadow/algorithm/ssd       算法示例/ssd通用检测
- Tron-shadow/algorithm/refinedet 算法示例/refinedet通用检测
- Tron-shadow/bj_run              bj_run五合一
- Tron-shadow/Landmark3D          人脸3d landmark
- Tron-shadow/Vsepa               Vsepa行人检测网络
- Tron-shadow/mixup               暴恐三模型
### 依赖配置

1. [下载第三方库](#第三方库)
2. [安装OpenCV-3.1.4-CUDA](#OpenCV-3.1.4-CUDA)


#### 下载第三方库

1. 请根据设备环境的CUDA以及CUDNN配置下载相应版本的第三方库，下载位置为项目主目录中

   - [3rdparty-tensorrt4.0.1.6-cuda9.0-cudnn7.1](http://pbv7wun2s.bkt.clouddn.com/3rdparty-tensorrt4.0.1.6-cuda9.0-cudnn7.1-radpidjson.zip)

   - [3rdparty-tensorrt4.0.1.6-cuda8.0-cudnn7.1](http://pbv7wun2s.bkt.clouddn.com/3rdparty-tensorrt4.0.1.6-cuda8.0-cudnn7.1-radpidjson.zip)


2. 解压文件夹

  ```Shell
  unzip 3rdparty-*.zip
  ```


#### 安装OpenCV-3.1.4-CUDA

1. 任意路径下载[opencv3.4.1](http://pbv7wun2s.bkt.clouddn.com/opencv-3.4.1.tar)

2. 在opencv3.4.1下载位置依次执行以下命令 

  ```Shell
  tar -xvf opencv-3.4.1.tar && rm opencv-3.4.1.tar
  cd opencv-3.4.1
  mkdir build 
  cd build
  cmake -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=/usr/local -D WITH_CUDA=ON -D ENABLE_FAST_MATH=ON -D CUDA_FAST_MATH=ON -D WITH_CUBLAS=1 -D WITH_NVCUVID=on -D CUDA_GENERATION=Auto ..
  make -j*
  make install
  ```
  
## caffe模型转trt相关流程
 
### flatten layer

修改type、flat_param

修改前
```
layer {
  name: "conv4_3_norm_mbox_loc_flat"
  type: "Flatten"
  bottom: "conv4_3_norm_mbox_loc_perm"
  top: "conv4_3_norm_mbox_loc_flat"
  flatten_param {
    axis: 1
  }
}
```

修改后
```
layer {
  name: "conv4_3_norm_mbox_loc_flat"
  type: "Reshape"
  bottom: "conv4_3_norm_mbox_loc_perm"
  top: "conv4_3_norm_mbox_loc_flat"
  reshape_param {
    shape {
      dim: 0
      dim: -1
      dim: 1
      dim: 1
    }
  }
}
```


### reshape layer

修改reshape_param

修改前
```
layer {
  name: "arm_conf_reshape"
  type: "Reshape"
  bottom: "arm_conf"
  top: "arm_conf_reshape"
  reshape_param {
    shape {
      dim: 0
      dim: -1
      dim: 2
    }
  }
}
```

修改后
```
layer {
  name: "arm_conf_reshape"
  type: "Reshape"
  bottom: "arm_conf"
  top: "arm_conf_reshape"
  reshape_param {
    shape {
      dim: 0
      dim: -1
      dim: 2
      dim:1
    }
  }
}
```

### norm layer

删掉norm_param字段，将layer name加入pluginfactory

修改前

```
layer {
  name: "conv5_3_norm"
  type: "Normalize"
  bottom: "conv5_3"
  top: "conv5_3_norm"
  norm_param {
    across_spatial: false
    scale_filler {
      type: "constant"
      value: 8.0
    }
    channel_shared: false
  }
}
```

修改后

```
layer {
  name: "conv5_3_norm"
  type: "Normalize"
  bottom: "conv5_3"
  top: "conv5_3_norm"
}
```

### priorbox

删除prior_box_param，修改pluginfactory相关代码：将layer name加入到pluginfactory里面，然后根据参数修改createPriorbox函数

修改前

```
layer {
  name: "conv4_3_norm_mbox_priorbox"
  type: "PriorBox"
  bottom: "conv4_3_norm"
  bottom: "data"
  top: "conv4_3_norm_mbox_priorbox"
  prior_box_param {
    min_size: 32.0
    aspect_ratio: 2.0
    flip: true
    clip: false
    variance: 0.10000000149
    variance: 0.10000000149
    variance: 0.20000000298
    variance: 0.20000000298
    step: 8.0
    offset: 0.5
  }
}
```

修改后

```
layer {
  name: "conv4_3_norm_mbox_priorbox"
  type: "PriorBox"
  bottom: "conv4_3_norm"
  bottom: "data"
  top: "conv4_3_norm_mbox_priorbox"
}
```

### detectionoutput

将这一层进行拆分。其参数部分在main.cpp和pluginfactory.cpp里的序列化和反序列化处进行修改

修改前

```
layer {
  name: "detection_out"
  type: "DetectionOutput"
  bottom: "odm_loc"
  bottom: "odm_conf_flatten"
  bottom: "arm_priorbox"
  bottom: "arm_conf_flatten"
  bottom: "arm_loc"
  top: "detection_out"
  include {
    phase: TEST
  }
  detection_output_param {
    num_classes: 29
    share_location: true
    background_label_id: 0
    nms_param {
      nms_threshold: 0.449999988079
      top_k: 200
    }
    code_type: CENTER_SIZE
    keep_top_k: 500
    confidence_threshold: 0.20000000298
    objectness_score: 0.00999999977648
  }
}
```

修改后

```
layer {
  name: "conf_data"
  type: "ApplyArmConf"
  bottom: "arm_conf_flatten"
  bottom: "odm_conf_flatten"
  top: "conf_data"
}
layer {
  name: "prior_data"
  type: "ApplyArmLoc"
  bottom: "arm_loc"
  bottom: "arm_priorbox"
  top: "prior_data"
}
```