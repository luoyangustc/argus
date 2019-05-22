#coding:utf-8
import crnn.crnn as crnn
import crnn.util as util
import torch
import torch.utils.data
import torchvision.transforms as transforms
from PIL import Image
from torch.autograd import Variable

from cfg import Config as cfg


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

    if cfg.PLATFORM == "GPU":
        model = crnn.CRNN(32, 1, len(alphabet) + 1, 256, 1).cuda()
        #model = torch.nn.DataParallel(model, device_ids=range(1))
        model.load_state_dict(torch.load(net_path))
    else:
        model = crnn.CRNN(32, 1, len(alphabet) + 1, 256, 1)
        model.load_state_dict(torch.load(net_path, map_location=lambda storage, loc: storage))

    return model,converter


def crnnRec_single(model,converter,im, use_Threshold=False):
   image = Image.fromarray(im).convert('L')
   #npimg = np.array(image)
   #if config.RECOGNITION.BINARY_IMAGE:
   #_, npimg = cv2.threshold(npimg, 127, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
   #if config.RECOGNITION.INVERT_COLOR_IMAGE:
   #image = Image.fromarray(255 - npimg)
   scale = image.size[1]*1.0 / 32
   w = image.size[0] / scale
   w = int(w)
   if w < 16:
        return ""

   transformer = resizeNormalize((w, 32))
   if cfg.PLATFORM == "GPU":
        image = transformer(image).cuda()
   else:
        image = transformer(image)
   image = image.view(1, *image.size())
   image = Variable(image)
   model.eval()
   preds = model(image, use_Threshold)
   vals, preds = preds.max(2)
   #preds = preds.squeeze(2)

   # if use_Threshold:
   #     vals = vals.squeeze(2)
   #     index = torch.ge(vals, config.RECOGNITION.THRESHOLD)
   #     preds = preds.mul(index.long())

   preds = preds.transpose(1, 0).contiguous().view(-1)
   preds_size = Variable(torch.IntTensor([preds.size(0)]))
   raw_pred = converter.decode(preds.data, preds_size.data, raw=True)
   sim_pred = converter.decode(preds.data, preds_size.data, raw=False)
   return sim_pred


