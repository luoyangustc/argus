class Config:
    PLATFORM = "GPU"
    TEST_GPU_ID = 0
    CLS_NET_DEF_FILE = 'models/deploy.prototxt'
    CLS_MODEL_PATH = 'models/text-classification-v0.2-t3.caffemodel'
    CLS_CONFIDENCE_THRESH = 0.6
