from easydict import EasyDict as edict

config = edict()

config.FEATURE_EXTRACTION = edict()
config.FEATURE_EXTRACTION.MODEL_PROTOTXT = 'models/extract_feature.prototxt'
config.FEATURE_EXTRACTION.MODEL_FILE = 'models/weight.caffemodel'
config.FEATURE_EXTRACTION.FEATURE_LAYER = 'pool5'