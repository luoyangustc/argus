if ! [[ -d mtcnn-caffe-model ]]; then
    wget http://p1ad6bkv9.bkt.clouddn.com/mtcnn-caffe-model.zip
    unzip mtcnn-caffe-model.zip
fi

if ! [[ -d feature-model ]]; then
    wget http://p3o9shm5o.bkt.clouddn.com/model-r34-amf-slim.zip
    unzip model-r34-amf-slim.zip

    mv model-r34-amf feature-model
fi