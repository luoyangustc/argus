#!/usr/bin/env bash
set -ex

APP=$1

ROOTDIR=$( cd "$( dirname "${BASH_SOURCE[0]}")" && cd ../.. && pwd )
cd $ROOTDIR

# export IMAGE=$(python -c "from aisdk.common.download_model import get_cfg_by_app; print(get_cfg_by_app('$APP')['image'])")

echo "run app $APP, dir $ROOTDIR, image $IMAGE..."

export PYTHONPATH=`pwd`/python:$PYTHONPATH
python -c "from aisdk.common.download_model import download_model; download_model('ava-facex-detect_tron-refinenet-mtcnn_20180824-private.tar')"
python -c "from aisdk.common.download_model import extract_model_tar; extract_model_tar('ava-facex-detect_tron-refinenet-mtcnn_20180824-private.tar')"
python -c "from aisdk.common.download_model import read_test_image_with_cache; print len(read_test_image_with_cache('serving/facex-detect/20180905/set20180905/0905_quality_orit.2.tsv'))"
python -c "from aisdk.common.download_model import parse_tsv_key, read_test_image_with_cache; print [len(read_test_image_with_cache('serving/facex-detect/set20180905/' + x[0])) for x in parse_tsv_key('serving/facex-detect/20180905/set20180905/0905_quality_orit.2.tsv')]"

if [ $CPU ];
then
    cd $ROOTDIR/cpp/src/app/face_detect && docker build -f cpu.dockerfile -t face_detect:cpu . && cd -

    docker run \
    -it \
    --rm \
    --net host \
    -v $ROOTDIR:/src \
    --name tmp-ai-sdk-test face_detect:cpu \
    sh /src/hack/face_detect/run.sh
else
    cd $ROOTDIR/cpp/src/app/face_detect && docker build -f gpu.dockerfile -t face_detect:gpu . && cd -

    docker run \
    -it \
    --rm \
    --net host \
    -e USE_DEVICE=GPU \
    --volume-driver nvidia-docker \
    --volume nvidia_driver_384.66:/usr/local/nvidia:ro \
    -v $ROOTDIR:/src \
    --device /dev/nvidiactl \
    --device /dev/nvidia-uvm \
    --device /dev/nvidia7 \
    --name tmp-ai-sdk-test face_detect:gpu \
    sh /src/hack/face_detect/run.sh
fi
