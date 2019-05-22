# MUST be imported firstly
import numpy as np

class Config:
    PLATFORM = "GPU"
    TEST_GPU_ID = 0

    DEBUG_SAVE_BOX_IMG = False

    DET_NET_DEF_FILE = 'models/deploy.prototxt'
    DET_MODEL_PATH = 'models/ctpn_trained_model.caffemodel'

    MEAN=np.float32([102.9801, 115.9465, 122.7717])
    SCALE=600
    MAX_SCALE=1000  #if the image is quite "long", MAX_SCALE can be set to a large value to avoid over-resize

    LINE_MIN_SCORE=0.7
    TEXT_PROPOSALS_MIN_SCORE=0.7
    TEXT_PROPOSALS_NMS_THRESH=0.3
    MAX_HORIZONTAL_GAP=50
    TEXT_LINE_NMS_THRESH=0.3
    MIN_NUM_PROPOSALS=2
    MIN_RATIO=1.2
    MIN_V_OVERLAPS=0.7
    MIN_SIZE_SIM=0.7
    TEXT_PROPOSALS_WIDTH=16

    DILATE_PIXEL = 10 # expand pixel for bbox shrink
    BREATH_PIXEL = 2 # expand pixel for final bbox, in case the first character is cut
    BINARY_THRESH = 150 # binary thresh for image binarization
