#!/usr/bin/env bash

python python/evals/main.py --help

echo '{
    "image_width": 224,
    "batch_size": 1,
    "use_device": "CPU",
    "tar_files": {
        "deploy.prototxt": "/tmp/eval/init/deploy.prototxt",
        "labels.csv": "/tmp/eval/init/labels.csv",
        "mean.binaryproto": "/tmp/eval/init/mean.binaryproto",
        "weight.caffemodel": "/tmp/eval/init/weight.caffemodel"
    }
}' > example.config.json

for file in `ls example.files`
do
    echo ">> EVAL >>", $file
    python python/evals/main.py eval caffe_classify \
    -c example.config.json \
    --request '[{"data": {"uri": "example.files/'$file'"}}]'
done