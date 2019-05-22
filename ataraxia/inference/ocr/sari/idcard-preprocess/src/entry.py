from src.IDCard.idcard import IDCard
import config
import cv2
import base64
import os

if __name__ == '__main__':
    f = IDCard()
    path = '/home/zhouzhao/Projects/pinganinvoice/test/身份证/idcard'
    for filename in os.listdir(path):
        image = cv2.imread(os.path.join(path,filename))
        byimage = base64.b64encode(cv2.imencode(".png", image.copy())[1].tostring())
        print(f.idcard_recog(byimage))