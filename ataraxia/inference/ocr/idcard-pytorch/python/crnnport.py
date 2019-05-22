#coding:utf-8
import torch
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import torch.utils.data
from torch.autograd import Variable
import os
import crnn.util as util
#import util
from PIL import Image
import crnn.crnn as crnn
#import crnn
import numpy as np
import cv2
from config import config

class resizeNormalize(object):

    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = size
        self.interpolation = interpolation
        self.toTensor = transforms.ToTensor()

    def __call__(self, img):
        img = img.resize(self.size, self.interpolation)
        img = self.toTensor(img)
        img.sub_(0.5).div_(0.5)
        return img

def crnnSource(net_path, alphabet):
    converter = util.strLabelConverter(alphabet)

    if config.PLATFORM == "GPU":
        model = crnn.CRNN(32, 1, len(alphabet) + 1, 256, 1).cuda()
        model.load_state_dict(torch.load(net_path))
    else:
        model = crnn.CRNN(32, 1, len(alphabet) + 1, 256, 1)
        model.load_state_dict(torch.load(net_path, map_location=lambda storage, loc: storage))
    #model.load_state_dict(torch.load(net_path))
    return model,converter


def crnnRec_single(model,converter,im, use_Threshold=False):
   image = Image.fromarray(im).convert('L')
   #npimg = np.array(image)
   #_, image = cv2.threshold(npimg, 127, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
   #image = Image.fromarray(255 - b_img)
   scale = image.size[1]*1.0 / 32
   w = image.size[0] / scale
   w = int(w)

   transformer = resizeNormalize((w, 32))
   if config.PLATFORM == "GPU":
        image = transformer(image).cuda()
   else:
        image = transformer(image)
   image = image.view(1, *image.size())
   image = Variable(image)
   
   model.eval()
   preds = model(image, use_Threshold)
   vals, preds = preds.max(2)
   #preds = preds.squeeze(2)

   if use_Threshold:
       #vals = vals.squeeze(2)
       index = torch.ge(vals, config.RECOGNITION.THRESHOLD)
       preds = preds.mul(index.long())

   preds = preds.transpose(1, 0).contiguous().view(-1)
   preds_size = Variable(torch.IntTensor([preds.size(0)]))
   raw_pred = converter.decode(preds.data, preds_size.data, raw=True)
   sim_pred = converter.decode(preds.data, preds_size.data, raw=False)
   return sim_pred

if __name__ == '__main__':
   address_model, address_converter = crnnSource(config.RECOGNITION.ADDRESS_MODEL_PATH,config.RECOGNITION.ADDRESS_ALPHABET)
   field_img = cv2.imread('address.jpg')
   field_predict = crnnRec_single(address_model, address_converter, field_img, use_Threshold=True)
   print(field_predict)

