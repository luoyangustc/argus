# Atlab Inference Frame
## 通用SDK推理框架，支持多bath输入，最高支持8张图像推理，底层采用tensorrt加速技术实现。

## For Example(限ubuntu16.04_cuda9.0_cudnn7.1.3 GPU P4卡环境下测试)
   - 下载范例测试图片和模型至当前目录解压
   [测试图像](http://p9s1ibz34.bkt.clouddn.com/face-detection-quality-test-images.zip),
   [测试模型](http://pbv7wun2s.bkt.clouddn.com/tron_fd_quality_ubuntu16.04_cuda9.0_cudnn7.1.3_engin_v0.0.2.tar)
   
   - 运行

   ```
    ./scripts/build_shell.sh
    ./build/tron/test_tron *
   ```

   - 多batch返回结果
    
    ************************************
    *  code   *   message   *   result *
    ************************************

    {
      {"code":200},{"message":tron_status_success},{"detections":[{"index":1,"score":0.9984379410743713,"class":"face","pts":[[43,306],[135,306],[135,373],[43,373]],"quality":"clear","orientation":"right","q_score":{"clear":0.9700742959976196,"blur":0.02766553685069084,"neg":0.0000025153385649900885,"cover":0.0021068176720291378,"pose":0.00015081478341016918}}]}
      {"code":200},{"message":tron_status_success},{"detections":[...]}
      {"code":200},{"message":tron_status_success},{"detections":[...]}
      ...
    }
   - 提PR合并后到[jenkins](https://jenkins.qiniu.io/view/ATLAB/view/AVA/job/ava-algorithm-base-build/build?delay=0sec)页面编绎镜像地址提供给serving组上线
