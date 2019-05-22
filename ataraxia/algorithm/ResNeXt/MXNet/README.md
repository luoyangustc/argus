
# ResNeXt - MXNet

## 执行步骤

### 1. 准备训练环境

#### Image准备

```bash
docker login -u ava-public@qiniu.com -p xxx
docker pull reg.qiniu.com/ava/resnext_mxnet_train
```

或者重新build

```bash
docker build . -f train/Dockerfile \
    -t ResNeXt_MXNet_Train \
    -t reg.qiniu.com/ava/resnext_mxnet_train
```

使用`AVA`，则将对应Image推送进`reg.qiniu.com`

```bash
docker push reg.qiniu.com/ava/resnext_mxnet_train
```

#### 代码准备

```bash
git clone -b dev --depth 1 https://github.com/qbox/ataraxia.git .
```

或者本地同步

```bash
rsync -e "ssh -i $HOME/.ssh/xxx -p 5989" \
    -v -r ${QBOX}/ataraxia/algorithm/ResNeXt \
    root@ssh.ava.atlab.ai:/workspace/mnt/group/${GROUP}/${USER}/
```

### 2. 准备数据

TODO 待补充

```bash
ls /workspace/mnt/group/ai-project/luoyang/bm_v41/recordio/train-256.rec
ls /workspace/mnt/group/ai-project/luoyang/bm_v41/recordio/dev-224.rec
```

### 3. 准备配置

参照`conf.yaml`

### 3. 训练

```bash
python ResNext/MXNet/train/train.py conf.yaml
```

### 4. 评估

```bash
```

### 5. 提交模型

```bash
```
