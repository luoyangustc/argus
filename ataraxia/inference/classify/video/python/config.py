from easydict import EasyDict as edict

config = edict()

config.FEATURE_CODING = edict()
config.FEATURE_CODING.MODEL_PREFIX = 'models/netvlad'
config.FEATURE_CODING.MODEL_EPOCH = 50
config.FEATURE_CODING.FEATURE_DIM = 2048
config.FEATURE_CODING.SYNSET='models/lsvc_class_index.txt'