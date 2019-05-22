# -*- coding: utf-8 -*-
### config file
from easydict import EasyDict as edict
import os
import crnn.keys as keys

config_dir = os.path.dirname(__file__)
print("config dir:%s" % config_dir)
config = edict()
config.PLATFORM = "GPU"
config.RECOGNITION = edict()
config.RECOGNITION.ADDRESS_ALPHABET = keys.alphabet
config.RECOGNITION.ADDRESS_MODEL_PATH = config_dir+'/models/netCRNN_0_37000.pth'  #models/netCRNN_v4_0_110000.pth
#config.RECOGNITION.ADDRESS_MODEL_PATH = 'models/netCRNN_v4_0_110000.pth'
config.RECOGNITION.NAME_ALPHABET = keys.alphabet
config.RECOGNITION.NAME_MODEL_PATH = config_dir+'/models/netCRNN_0_37000.pth'   # 'models/netCRNN_v4_0_110000.pth'
#config.RECOGNITION.DIGITS_MODEL_PATH = 'models/idcard_digits.pth' #idcard_digits.pth' #'models/digits_rec.params'
config.RECOGNITION.DIGITS_MODEL_PATH = config_dir+'/models/digits_new_model67000.pth'
config.RECOGNITION.DIGITS_RECOG_THRESH = 0.5 #origin = 0.8
config.RECOGNITION.BINARY_THRESHOLD = 100
config.RECOGNITION.DIGITS_MIN_BIT = 10
config.RECOGNITION.DIGITS_MAX_BIT = 18
config.RECOGNITION.THRESHOLD=0.4
config.RECOGNITION.POSTTHRESHOLD=1

config.SEGMENT = edict()
config.SEGMENT.TEMPLATE_FIELDS_LIST = ["name","address1","photo","month","year","nation","id","day","gender",
									  "address2","address3"]
config.SEGMENT.TEMPLATE_FIELDS = {"name": [[138, 35], [340, 35], [340, 76],[138, 76]],
								  "address1":[[151,266], [555,266],[555,299], [151,299]],
								  "photo":[[564,42], [900,42],  [900,414],[564,414]],
								  "month":[[301,189], [348,189], [348,222], [301,222]],
								  "year":[[143,189], [251,189],[251,221], [143,221]],
								  "nation":[[342,116],[449,116],[449,149],[342,149]],
								  "id":[[294,447], [830,447],[830,494], [294,494]],
								  "day":[[382,188],[439,188],[439,220],[382,220]],
								  "gender":[[137,115],[246,115],[246,150],[137,150]],
								  "address2":[[150,301],[555,301],[555,349],[150,349]],
								  "address3":[[151,350],[553,350],[350,398],[151,398]]}
config.SEGMENT.TEMPLATE_IMG = config_dir+"/template/idcard_template.png"
config.SEGMENT.POST_MATCH= edict()
config.SEGMENT.POST_MATCH.threshold = 0.9
config.SEGMENT.POST_MATCH.good_match_num = 10
config.SEGMENT.POST_MATCH.knn_match_k = 2

config.POST = edict()
config.POST.GENDER_FILTER_MALE = ['男', '勇', '剃', '嘹', '舅']
config.POST.GENDER_FILTER_FEMALE = ['女', '太']
