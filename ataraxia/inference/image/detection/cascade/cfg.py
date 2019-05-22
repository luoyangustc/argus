class Config:
    PLATFORM = "GPU"
    TEST_GPU_ID = 1
    CLS_NET_DEF_FILE = '../models/deploy.prototxt'
    CLS_MODEL_PATH = '../models/weight.caffemodel'
    CLS_LABEL_INDEX = '../models/labels.csv'
