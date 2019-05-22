from torch.autograd import Variable
from . import dataset, py_util
from . import util
from PIL import Image
import torch
import math
# import logging
import config
import cv2


class BaseModel(object):
    """docstring for BaseModel"""

    def __init__(self):
        super(BaseModel, self).__init__()
        self.model = None
        self.alphabet = None
        self.converter = None

    def im_predict(self, im):
        predict = self.model(im)
        predict_len = Variable(torch.IntTensor([predict.size(0)]))
        _, acc = predict.max(2)
        if int(torch.__version__.split('.')[1]) < 2:
            acc = acc.squeeze(2)
        acc = acc.transpose(1, 0).contiguous().view(-1)
        sim_preds = self.converter.decode(
            acc.data, predict_len.data, raw=False)
        return sim_preds

    def mutil_predict(self, imgs):
        batch_size = imgs.shape[0]
        predict = self.model(imgs)
        predict_len = Variable(torch.IntTensor([predict.size(0)] * batch_size))
        _, acc = predict.max(2)
        if int(torch.__version__.split('.')[1]) < 2:
            acc = acc.squeeze(2)
        acc = acc.transpose(1, 0).contiguous().view(-1)
        sim_preds = self.converter.decode(
            acc.data, predict_len.data, raw=False)
        return sim_preds

    def deploy(self, image_list, isbatch=False):
        # print(self.alphabet)
        self.converter = util.strLabelConverter(self.alphabet)
        # model = crnn
        self.model.eval()
        if not isbatch:
            res = []
            for im in image_list:
                img = Image.fromarray(im)
                w, h = img.size
                imgH = 32
                # imgW = int(imgH * w / h)
                # transform = dataset.resizeNormalize(imgW, imgH)
                transform = dataset.resizeNormalizePadding(imgH)
                im = transform(img)
                im = im.view(1, im.size()[0], im.size()[1], im.size()[2])
                if(config.USE_GPU):
                    preds = self.im_predict(Variable(im.cuda().half()))
                else:
                    preds = self.im_predict(Variable(im))

                # logging.critical(preds)
                res.append(preds)
        else:
            res = ['' for i in range(len(image_list))]
            idx = 1
            for im in image_list:
                cv2.imwrite('./disp/' + str(idx) + '.jpg', im)
                idx += 1
            ctrl = BatchCtrl(image_list)
            batch, idxs = ctrl.get_batch()
            while batch is not None:
                # logging.critical(batch.size)
                if(config.USE_GPU):
                    preds = self.mutil_predict(Variable(batch.cuda().half()))
                else:
                    preds = self.mutil_predict(Variable(batch))
                # logging.critical(preds)
                if isinstance(preds, str):
                    preds = [preds]
                for i, pred in enumerate(preds):
                    res[idxs[i]] = pred
                batch, idxs = ctrl.get_batch()
            # logging.critical(res)
        return res


class BatchCtrl():
    def __init__(self, image_list, memory=100000):
        self.memory = memory
        self.image_list = []
        self.image_width = []
        self.image_num = len(image_list)
        self.imgH = 32
        for i, im in enumerate(image_list):
            img = Image.fromarray(im)
            w, h = img.size
            imgW = int(self.imgH * w / h)
            self.image_list.append(img)
            self.image_width.append((i, imgW))
        self.image_width = sorted(
            self.image_width, key=lambda x: x[1], reverse=True)

    def get_batch(self):
        if self.image_width == []:
            return None, None
        batch = []
        idxs = []
        maxW = self.image_width[0][1]
        max_num = math.floor(self.memory / maxW)
        transform = dataset.resizeNormalize(maxW, self.imgH)
        for i in range(max_num):
            if i == len(self.image_width):
                break
            img_idx = self.image_width[i][0]
            im = transform(self.image_list[img_idx])
            batch.append(im)
            idxs.append(img_idx)

        batch = torch.stack(batch, 0)
        self.image_width = self.image_width[max_num:]

        return batch, idxs
