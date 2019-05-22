import os
import sys
sys.path.insert(0,'.')
from src.zengzhishui.zengzhishui import ZenZhuiShui_reco
import cv2
import config
import json

path = '/home/zhouzhao/Projects/pinganinvoice/test/增值税发票/image'
for imagename in os.listdir(path):
    image = cv2.imread(os.path.join(path,imagename))
    handle_zengzhishui = ZenZhuiShui_reco(config.MODEL_CRANN_ZENGZHISHUI_API)
    res = handle_zengzhishui.predictAndGenXML(image)
    print(json.loads(res))