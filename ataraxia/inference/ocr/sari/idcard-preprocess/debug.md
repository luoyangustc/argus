export RUN_MODE=standalone

cd python/evals
wget http://p9zv90cqq.bkt.clouddn.com/001.jpg http://p9zv90cqq.bkt.clouddn.com/001_back.jpg


import eval
import json
import config
import time
import base64
from evals.src.idcard_class import IDCARDCLASS
import config
import cv2
import sys
import numpy as np
from evals.src.IDCardfront.Aligner import AlignerIDCard
from evals.src.IDCardback.Aligner import AlignerIDCardBack


idcardcls = IDCARDCLASS(config.ALIGNER_TEMPLATE_IDCARDCLASS_IMG_PATH, config.ALIGNER_TEMPLATE_IDCARDCLASSBACK_IMG_PATH)
idcardfront = AlignerIDCard(config.ALIGNER_TEMPLATE_IDCARD_IMG_PATH, config.ALIGNER_TEMPLATE_IDCARD_LABEL_PATH)
idcardback = AlignerIDCardBack(config.ALIGNER_TEMPLATE_IDCARDBACK_IMG_PATH, config.ALIGNER_TEMPLATE_IDCARDBACK_LABEL_PATH)

img = cv2.imread('001.jpg')
cls = idcardcls.run(img)
alignedImg,names,regions,boxes = idcardfront.predet(img)

reqs = [{"data": {"uri": 'data:application/octet-stream;base64,' + base64.b64encode(cv2.imencode('.jpg',alignedImg)[1]),"body":''}}]
reqs[0]["params"] = {
    "type": "prerecog",
    "class": cls,
    "img": base64.b64encode(cv2.imencode('.jpg',alignedImg)[1]),
    "names": names,
    "regions": regions,
    "bboxes": boxes,
    "detectedBoxes": [[[121, 231], [413, 229], [413, 260], [121, 263]], [[72, 173], [346, 176], [345, 205], [72, 202]], [[45, 62], [215, 60], [215, 93], [46, 95]], [[243, 374], [619, 371], [620, 404], [244, 407]], [[131, 270], [239, 270], [239, 299], [131, 298]], [[46, 374], [205, 373], [206, 403], [46, 404]], [[47, 122], [162, 120], [162, 150], [47, 151]], [[206, 121], [302, 121], [302, 148], [206, 148]], [[44, 232], [119, 233], [118, 259], [44, 258]]],
    "texts": ['\xe6\xb2\xb3\xe5\x8d\x97\xe7\x9c\x81\xe9\xa1\xb9\xe5\x9f\x8e\xe5\xb8\x82\xe8\x8a\x99\xe8\x93\x89\xe5\xb7\xb7\xe4\xb8\x9c\xe5\x9b\x9b', '\xe8\x83\xa1\xe5\x90\x8c2\xe5\x8f\xb7', '\xe6\x80\xa7\xe5\x88\xab\xe2\x80\x98\xe5\xa5\xb3\xe4\xba\xba\xe6\xb0\x91\xe2\x80\x98\xe6\x97\x8f\xe6\xb1\x89', '412702199705127504', '1997\xe5\xb9\xb45\xe6\x9c\x8812\xe6\x97\xa5', '\xe5\xbc\xa0\xe6\x9d\xb0']
}

img2=cv2.imdecode(np.fromstring(base64.b64decode(reqs[0]["params"]["img"]), dtype=np.uint8), 1)
boxes = idcardfront.prerecog(reqs[0]["params"]["detectedBoxes"],img2,reqs[0]["params"]["names"],reqs[0]["params"]["regions"],reqs[0]["params"]["bboxes"])
reqs[0]["params"]['bboxes'] = boxes

res = idcardfront.postprocess(reqs[0]["params"]["bboxes"],reqs[0]["params"]["texts"],reqs[0]["params"]["regions"],reqs[0]["params"]["names"])

<!-- ret = eval.prerecog(idcardfront, 1, reqs[0]) -->
<!-- ret = eval.postprocess(idcardfront, 1, reqs[0]) -->



model = eval.create_net({"batch_size":1})
image = cv2.imread('005Zu0d2ly1fliuvdil2uj30hr1k1do7.jpg')
bboxes = [[15,888,564,894,563,933,14,926],[22,1788,582,1792,582,1833,21,1829],[19,1238,551,1242,551,1281,19,1277],[19,1688,530,1694,530,1732,19,1726],[4,1887,589,1892,589,1932,4,1928],[6,590,575,585,575,627,7,631],[13,838,595,841,595,881,13,878],[18,289,527,292,526,330,17,326],[22,1341,588,1338,588,1377,22,1381],[65,1038,518,1043,518,1080,65,1075],[23,1193,572,1188,573,1227,24,1232],[16,689,532,693,532,732,15,728],[23,637,562,641,562,681,23,676],[16,186,591,192,591,233,16,228],[28,1136,590,1140,589,1181,28,1177],[72,988,526,993,526,1031,71,1027],[8,537,574,542,573,582,8,577],[4,1388,588,1392,587,1432,4,1428],[14,437,331,440,330,483,13,479],[9,237,588,241,587,280,9,276],[29,1441,523,1438,523,1477,30,1480],[18,1591,589,1588,589,1628,18,1631],[23,1490,396,1492,395,1532,23,1529],[6,1637,566,1641,565,1681,5,1677],[29,1940,348,1945,347,1981,28,1976],[13,338,436,341,435,380,13,377],[5,1840,588,1844,588,1883,4,1879],[23,787,586,790,586,831,22,827]]
bboxes = list(map(lambda x:[[x[0],x[1]],[x[2],x[3]],[x[4],x[5]],[x[6],x[7]]],bboxes))
crann_recog = model[0]['crann_recog']

imglist = crann_recog.cutimagezz(image,bboxes)
predictions, text_bboxes = crann_recog.deploy(imglist)


curl -v -X POST -H "Authorization:QiniuStub uid=1&ut=2" -H "Content-Type:application/json" 127.0.0.1:43660/v1/eval -d '{ "data":{ "uri":"http://p24v9nypo.bkt.clouddn.com/005Zu0d2ly1fliuvdil2uj30hr1k1do7.jpg" }, "params": { "bboxes":[[15,888,564,894,563,933,14,926],[22,1788,582,1792,582,1833,21,1829],[19,1238,551,1242,551,1281,19,1277],[19,1688,530,1694,530,1732,19,1726],[4,1887,589,1892,589,1932,4,1928],[6,590,575,585,575,627,7,631],[13,838,595,841,595,881,13,878],[18,289,527,292,526,330,17,326],[22,1341,588,1338,588,1377,22,1381],[65,1038,518,1043,518,1080,65,1075],[23,1193,572,1188,573,1227,24,1232],[16,689,532,693,532,732,15,728],[23,637,562,641,562,681,23,676],[16,186,591,192,591,233,16,228],[28,1136,590,1140,589,1181,28,1177],[72,988,526,993,526,1031,71,1027],[8,537,574,542,573,582,8,577],[4,1388,588,1392,587,1432,4,1428],[14,437,331,440,330,483,13,479],[9,237,588,241,587,280,9,276],[29,1441,523,1438,523,1477,30,1480],[18,1591,589,1588,589,1628,18,1631],[23,1490,396,1492,395,1532,23,1529],[6,1637,566,1641,565,1681,5,1677],[29,1940,348,1945,347,1981,28,1976],[13,338,436,341,435,380,13,377],[5,1840,588,1844,588,1883,4,1879],[23,787,586,790,586,831,22,827]]}}'