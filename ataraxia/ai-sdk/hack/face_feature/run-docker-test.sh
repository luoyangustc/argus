#!/usr/bin/env bash
set -ex

APP=$1

ROOTDIR=$( cd "$( dirname "${BASH_SOURCE[0]}")" && cd ../.. && pwd )
cd $ROOTDIR

# export IMAGE=$(python -c "from aisdk.common.download_model import get_cfg_by_app; print(get_cfg_by_app('$APP')['image'])")

echo "run app $APP, dir $ROOTDIR, image $IMAGE..."

python -c "from aisdk.common.download_model import download_model; download_model('ava-facex-feature-v3/caffe-mxnet/201803152400.tar')"

if [ $CPU ];
then
    export IMAGE="ff:cpu"
    docker run \
    -it \
    --rm \
    --net host \
    -v $ROOTDIR:/src \
    --name tmp-ai-sdk-test $IMAGE \
    /src/app/face_feature_v3/examples/runtest.sh
else
    export IMAGE="reg.qiniu.com/avaprd/ava-eval-face-feature:v4-201803301958--20180509-v80-dev"
    docker run \
    -it \
    --rm \
    --net host \
    -e USE_DEVICE=GPU \
    --volume-driver nvidia-docker \
    --volume nvidia_driver_375.26:/usr/local/nvidia:ro \
    -v $ROOTDIR:/src \
    --device /dev/nvidiactl \
    --device /dev/nvidia-uvm \
    --device /dev/nvidia1 \
    --name tmp-ai-sdk-test $IMAGE \
    /src/app/face_feature_v3/examples/runtest.sh
fi