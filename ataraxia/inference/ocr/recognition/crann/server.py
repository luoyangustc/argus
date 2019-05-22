#!flask/bin/python
from flask import Flask
from flask import request
import json
from src.crannRec.crannrec import CrannRecModel
import cv2
import numpy as np
import base64
import config
import time
import logging

app = Flask(__name__)

handler_crann_rec = None
showtime = False

def initimglist(data):
    imglist = [cv2.imdecode(np.fromstring(base64.b64decode(img.encode(
        'utf-8')), dtype=np.uint8), 1) for img in json.loads(data.decode('utf-8'))]
    imglist = [255 - cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in imglist]
    return imglist

@app.route('/')
def index():
    return "Hello, World!"

# ================================================
# 稳定api - 通用api
# ================================================

@app.route('/base/recognize/boxescrann', methods=['POST'])
def crannrecboxes():
    start_cut = time.time()
    imglist = handler_crann_rec.cutimagezz(request.data.decode('utf-8'))
    end_cutimg = time.time()
    res = handler_crann_rec.deploy(imglist)
    end_deploy = time.time()
    result = json.dumps(res)
    return result

@app.route('/base/recognize/crann', methods=['POST'])
def crannrec():
    imglist = initimglist(request.data)
    res = handler_crann_rec.deploy(imglist,True)
    res = json.dumps(res)
    return res





# ================================================
# 稳定api - 定制api
# ================================================

@app.route('/base/recognize/drivercrann', methods=['POST'])
def driver():
    start_cut = time.time()
    imglist = handler_driver.cutimage(request.data.decode('utf-8'))
    end_cutimg = time.time()
    res = handler_driver.deploy(imglist)
    end_deploy = time.time()
    result = json.dumps(res)
    return result

@app.route('/base/recognize/classifycrann', methods=['POST'])
def crannclass():
    imglist = handler_crann_rec.cutimage(request.data.decode('utf-8'),isurl = True)
    res = handler_crann_rec.deploy(imglist)
    result = json.dumps(res)
    return result

@app.route('/base/recognize/yingyezhizhaocrann', methods=['POST'])
def crannyyzz():
    imglist = initimglist(request.data)
    res = handler_yyzz.deploy(imglist,True)
    res = json.dumps(res)
    return res








# ================================================
# 测试api
# ================================================

@app.route('/test/recognize/test', methods=['POST'])
def test():
    pass


if __name__ == '__main__':
    # init model
    handler_yyzz = CrannRecModel(config.MODELFOLE_CRANN_YYZZ_URL, config.MODELFOLE_CRANN_YYZZ_YML_URL)
    handler_crann_rec = CrannRecModel(
        config.MODELFILE_CRANN_URL, config.MODELFILE_CRANN_YML_URL)
    handler_driver = CrannRecModel(config.MODELFILE_CRANN_DRIVER_URL, config.MODELFILE_CRANN_YML_URL)


    app.run(host=config.HOST, port=config.PORT, debug=False, threaded = True) #threaded = True , processes = 5
