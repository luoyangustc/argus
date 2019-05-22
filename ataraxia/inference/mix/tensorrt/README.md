# Atlab Inference Frame 还原卡项目
## 通用SDK推理框架，支持多bath输入，最高支持16张图像推理，底层采用tensorrt加速技术实现。

## For Example(限ubuntu16.04_cuda9.0_cudnn7.1.3 GPU P4卡环境下测试)
   - 下载范例测试图片和模型至当前目录解压
   [测试图像](http://p9s1ibz34.bkt.clouddn.com/face-detection-quality-test-images.zip),
   [测试模型](http://pbv7wun2s.bkt.clouddn.com/tron_bj_run_ubuntu16.04_cuda9.0_cudnn7.1.3_engin_v0.0.2.tar)
   
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
      {"code":200},{"message":tron_status_success},{"result":{"march":0,"normal":0,"text":0,"face":1,"bk":1,"pulp":0,"facenum":1,"faces":[{"pts":[[670,314],[807,314],[807,487],[670,487]],"features":[1.9563690423965455,1.2681313753128052,...]}]}}
      {"code":200},{"message":tron_status_success},{"result":{"march":1,"normal":0,"text":0,"face":0,"bk":1,"pulp":0,"facenum":0,"faces":[]}}
      {"code":200},{"message":tron_status_success},{"result":{"march":0,"normal":0,"text":0,"face":0,"bk":1,"pulp":0,"facenum":0,"faces":[]}}
      ...
    }
   - 提PR合并后到[jenkins](https://jenkins.qiniu.io/view/ATLAB/view/AVA/job/ava-algorithm-base-build/build?delay=0sec)页面编绎镜像地址提供给serving组上线
