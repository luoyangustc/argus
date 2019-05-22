# MUST be imported firstly
import sys
sys.path.insert(0, "./src")
import crnn.keys as keys

class Config:
    PLATFORM = "GPU"
    TEST_GPU_ID = 0

    RECOG_MODEL_PATH = 'models/netCRNN_v4_0_110000.pth'
    TEXT_RECOG_ALPHABET = keys.alphabet

    BLOG_LINE_HEIGHT_DIFF_IN_PERCENT = (0.5,1.5) #  be connected if the height of adjacent line is alike