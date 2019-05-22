"""in mnist, see more explanation at http://mxnet.io/tutorials/python/mnist.html
"""
import numpy as np
import cv2
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from config import config

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5)
        self.conv2 = nn.Conv2d(20, 50, 5)
        self.fc1   = nn.Linear(800, 500)
        self.fc2   = nn.Linear(500, 11)
        self.softmax = nn.Softmax()

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = self.fc2(out)
        out = self.softmax(out)
        return out

class resizeNormalize(object):

    def __init__(self):
        self.toTensor = transforms.ToTensor()

    def resize_image(self, img, target_size):
        im_shape = img.shape
        im_size_min = np.min(im_shape[0:2])
        im_size_max = np.max(im_shape[0:2])
        im_scale = float(target_size[0]) / float(im_size_max)

        img = cv2.resize(img, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_LINEAR)

        # pad to product of stride
        padded_im = np.zeros(target_size)
        padded_im[:img.shape[0], :img.shape[1]] = img
        return padded_im


    def __call__(self, img):
        img = self.resize_image(img, (28,28))
        img = np.expand_dims(img, axis=-1)
        img = self.toTensor(img)
        img.sub_(0.5).div_(0.5)
        return img

class digitDetect(object):
    def __init__(self, model_path, digits_min_bit, digits_max_bit):
        if config.PLATFORM == "GPU":
            self.digit_model = LeNet().cuda()
            self.digit_model.load_state_dict(torch.load(model_path))
        else:
            self.digit_model = LeNet()
            self.digit_model.load_state_dict(torch.load(model_path, map_location=lambda storage, loc: storage))

        self.transform = resizeNormalize()
        self.DIGITS_MIN_BIT = digits_min_bit
        self.DIGITS_MAX_BIT = digits_max_bit


    def findContours(self, img, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE):
        if cv2.__version__[0] == '2':
            contours2, hierarchy2 = cv2.findContours(img.copy(), mode, method)
        elif cv2.__version__[0] == '3':
            _, contours2, hierarchy2 = cv2.findContours(img.copy(), mode, method)
        return contours2, hierarchy2


    def read_data(self,img_bgr):
        """
        download and read data into numpy
        - input:
            - imgpath
        - output:
            - image_chips: each digit in one image_chip
            - num_image_chips: how many digits are detected in the image. Detection is based on contour blur and detection.
        """
        image_chips = []

        img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        _, img = cv2.threshold(img_gray, config.RECOGNITION.BINARY_THRESHOLD, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        img = 255 - img  # change bg to black

        # find digits contours
        kx = 3; ky = 3
        img = cv2.GaussianBlur(img, (kx, ky), 0)
        contours, _ = self.findContours(img)
        rects = []; sort_m = [] # sort_m is used for sort rects by their position
        for i, contour in enumerate(contours):
            x0, y0, x1, y1 = np.min(contour[:, :, 0]), np.min(contour[:, :, 1]), np.max(contour[:, :, 0]), np.max(
                contour[:, :, 1])
            rects.append([x0, y0, x1, y1])
            sort_m.append(x0)

        sorted_index = sorted(range(len(sort_m)), key=lambda k: sort_m[k])
        rects_sorted = []
        for idx in sorted_index:
            rects_sorted.append(rects[idx])

        # segment image to image_chips
        for i, rect in enumerate(rects_sorted):
            x0, y0, x1, y1 = rect
            image_chip = img[y0:y1, x0:x1]
            image_chips.append(image_chip)
            #cv2.imwrite('image_chip_{}.jpg'.format(str(i)), image_chip)

        image_chips = [self.transform(image) for image in image_chips]
        image_chips = torch.cat([t.unsqueeze(0) for t in image_chips], 0)
        return image_chips



    def probs_to_idnumber(self, _probs):
        id_num = ''
        probs = []
        for prob in _probs:
            max_prob = prob.data.max(-1)[0][0]
            max_index = prob.data.max(-1)[1][0]
            if max_prob < config.RECOGNITION.DIGITS_RECOG_THRESH:
                continue
            probs.append(max_prob)
            if max_index == 10:
                id_num += 'X'
            else:
                id_num += str(max_index)

        return id_num, probs


    def threshold_digits(self, digits, probs):
        if len(digits) < self.DIGITS_MIN_BIT:
            digits_keep = ""
            probs_keep = []
            return digits_keep, probs_keep
        elif len(digits) > self.DIGITS_MAX_BIT:
            sorted_index = sorted(range(len(probs)), key=lambda k: probs[k], reverse = True)
            idx_keep = sorted_index[:self.DIGITS_MAX_BIT]
            digits_keep = ""
            probs_keep = []
            for i in range(len(digits)):
                if i in idx_keep:
                    digits_keep += digits[i]
                    probs_keep.append(probs[i])
            return digits_keep, probs_keep
        else:
            return digits, probs


    def digits_predict(self, img):
        image_chips = self.read_data(img)
        if config.PLATFORM == "GPU":
            data = Variable(image_chips.cuda(), volatile=True)
        else:
            data = Variable(image_chips, volatile=True)
        probs = self.digit_model.forward(data)
        digits, probs = self.probs_to_idnumber(probs)
        digits, probs = self.threshold_digits(digits, probs)
        return digits, probs


if __name__ == '__main__':
    # parse args
    parser = argparse.ArgumentParser(description="test mnist",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--img_path', type=str, help='test_image or folder of test images')
    parser.add_argument('--model_path', default='models/idcard_digits.pth',type=str)

    args = parser.parse_args()

    digits_model = digitDetect(args.model_path, 5,18)
    if os.path.isdir(args.img_path):
        for parent, dirnames, filenames in os.walk(args.img_path):
            for idx, filename in enumerate(filenames):
                if '.jpg' in filename:
                    filepath = os.path.join(parent, filename)
                    img = cv2.imread(filepath, cv2.IMREAD_COLOR)
                    id_num, probs = digits_model.digits_predict(img)
                    print(filename, id_num)
    else:
        img = cv2.imread(args.img_path, cv2.IMREAD_COLOR)
        id_num, probs = digits_model.digits_predict(img)
        print(args.img_path, id_num)
