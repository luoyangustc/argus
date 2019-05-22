#!/usr/bin/env bash

echo '{
    "eval": {
        "image_width": 224,
        "batch_size": 1,
        "use_device": "CPU",
        "tar_files": {
            "deploy.prototxt": "/tmp/eval/init/deploy.prototxt",
            "labels.csv": "/tmp/eval/init/labels.csv",
            "mean.binaryproto": "/tmp/eval/init/mean.binaryproto",
            "weight.caffemodel": "/tmp/eval/init/weight.caffemodel"
        }
    }
}' > example.config.json

./hack-embed-eval -f example.config.json /workspace/serving/example.files