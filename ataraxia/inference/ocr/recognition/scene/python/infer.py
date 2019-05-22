from __future__ import print_function
import argparse
import os
import torchvision.transforms as transforms
import cv2
import numpy as np
from PIL import Image
from math import sin, fabs, radians, cos, atan2, degrees
from torch.autograd import Variable
import torch.utils.data
from config import CONFIG
import json
import dataset
import utils.util as util
import datetime
import crnn.crnn as crnn
import copy


class TextRecognizer():
    def __init__(self, model_path=CONFIG.model_path, alphabet=CONFIG.alphabet):

        self.model = crnn.CRNN(n_class=len(alphabet) + 1)
        self.model.load_state_dict(torch.load(model_path)['state_dict'])
        self.converter = util.strLabelConverter(alphabet)
        self.toTensor = transforms.ToTensor()
        self.model = self.model.cuda()
        self.model.eval()

    def rotate_image(self, img, degree, pt1, pt2, pt3, pt4):
        height, width = img.shape[:2]
        new_height = int(width * fabs(sin(radians(degree))) +
                         height * fabs(cos(radians(degree))))
        new_width = int(height * fabs(sin(radians(degree))) +
                        width * fabs(cos(radians(degree))))
        mat_rotation = cv2.getRotationMatrix2D(
            (width / 2, height / 2), degree, 1)
        mat_rotation[0, 2] += (new_width - width) / 2
        mat_rotation[1, 2] += (new_height - height) / 2
        img_rotation = cv2.warpAffine(
            img, mat_rotation, (new_width, new_height), borderValue=(255, 255, 255))
        pt1 = list(pt1)
        pt3 = list(pt3)

        [[pt1[0]], [pt1[1]]] = np.dot(
            mat_rotation, np.array([[pt1[0]], [pt1[1]], [1]]))
        [[pt3[0]], [pt3[1]]] = np.dot(
            mat_rotation, np.array([[pt3[0]], [pt3[1]], [1]]))
        img_out = img_rotation[int(pt1[1]):int(
            pt3[1]), int(pt1[0]):int(pt3[0])]
        height, width = img_out.shape[:2]
        return img_out

    def resize_img(self, img):
        if img is None or img.shape[0] == 0 or img.shape[1] == 0:
            return None, 0
        if img.shape[0] > img.shape[1] * 1.8:
            img = np.rot90(img)

        scale = float(img.shape[0]) / 32.0

        w = int(float(img.shape[1]) / scale)

        if w < 16:
            w = 32

        img = cv2.resize(img, (w, 32))

        return img, w

    def regular_pts(self, pt1, pt2, pt3, pt4, max_width):
        if pt1[1] != pt2[1] and abs(pt1[1] - pt2[1]) <= 3:
            pt1[1] = pt2[1] = min(pt1[1], pt2[1]) + 1

        if pt3[1] != pt4[1] and abs(pt3[1] - pt4[1]) <= 3:
            pt3[1] = pt4[1] = max(pt3[1], pt4[1]) - 1

        if pt1[0] != pt4[0] and abs(pt1[0] - pt4[0]) <= 3:
            pt1[0] = pt4[0] = min(pt1[0], pt4[0])

        if pt2[0] != pt3[0] and abs(pt2[0] - pt3[0]) <= 3:
            pt2[0] = pt3[0] = max(pt2[0], pt3[0])
        if pt1[1] == pt2[1] and pt3[1] == pt4[1] and 0 < abs(pt4[1] - pt1[1]) < abs(pt3[0] - pt4[0]):
            pt1[0] = max(0, pt1[0] - 5)
            pt4[0] = max(0, pt4[0] - 5)
            pt2[0] = min(max_width - 1, pt2[0] + 5)
            pt3[0] = min(max_width - 1, pt3[0] + 5)

        return pt1, pt2, pt3, pt4

    def infer(self, src_img, text_rects):
        if src_img is None:
            return None
        texts = []
        img_tups = []
        ids = 0
        for rect in text_rects:
            pt1 = copy.copy(rect[0])
            pt2 = copy.copy(rect[1])
            pt3 = copy.copy(rect[2])
            pt4 = copy.copy(rect[3])
            pt1, pt2, pt3, pt4 = self.regular_pts(
                pt1, pt2, pt3, pt4, src_img.shape[1])

            part_img = self.rotate_image(src_img, degrees(
                atan2(pt2[1] - pt1[1], pt2[0] - pt1[0])), pt1, pt2, pt3, pt4)

            img, width = self.resize_img(part_img)
            img_tups.append((img, width, ids))
            ids += 1
        if not img_tups:
            return texts
        sorted_imgs = sorted(
            img_tups, key=lambda img_width: img_width[1])
        _max = 280 * 32
        batchs = []
        idx = []
        widths = []

        new_start = -1
        for i, data in enumerate(sorted_imgs):

            if data[1] != 0:
                new_start = i
                break
            texts.append("")
        if new_start < 0:
            return texts
        start = new_start

        for i, data in enumerate(sorted_imgs):
            idx.append(data[2])
            widths.append(data[1])
            if (i - start) * data[1] > _max:
                batchs.append((start, i))
                start = i
        else:
            batchs.append((start, len(sorted_imgs)))

        for start, end in batchs:
            max_w = sorted_imgs[end - 1][1]
            trans = dataset.resizeNormalize(max_w)
            imgs = [trans(im, w) for im, w, _ in sorted_imgs[start: end]]
            imgs = torch.cat([im.unsqueeze(0) for im in imgs], 0)
            bsz = imgs.size(0)
            imgs = Variable(imgs.cuda())
            preds = self.model(imgs)

            predict_len = Variable(torch.IntTensor([preds.size(0)] * bsz))
            _, preds = preds.max(2)

            # preds = preds.squeeze(2)
            preds = preds.transpose(1, 0).contiguous().view(-1)
            sim_preds = self.converter.decode(
                preds.data, predict_len.data, raw=False)
            if bsz == 1:
                texts.append(sim_preds)
            for pred in sim_preds:
                texts.append(pred)
        texts_idx = zip(texts, idx)

        texts_idx.sort(key=lambda data: data[1])
        texts, idx = zip(*texts_idx)
        return texts


def parse_args():
    """
    Parse input arguments
    """

    parser = argparse.ArgumentParser(description='infer the crnn')
    parser.add_argument('--image_folder', dest='image_folder', help='infer which folder images', default="./images",
                        type=str)
    parser.add_argument('--is_output', dest='is_output', help='if you will output the result to files', type=bool,
                        default=True)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    recognizer = TextRecognizer()

    args = parse_args()
    result = []
    with open('file/lsc_bk/label.json', encoding="utf8") as f:
        datas = json.load(f)
    for data in datas:
        img_path = data["url"]
        rects = []
        labels = []
        for d in data["data"]:
            print("dhere:", d)
            if "content" in d and d["content"] != "a" and d["content"] != "":
                bbox = d["bbox"]
                rect = [bbox[0][0], bbox[0][1], bbox[1][0], bbox[1][1],
                        bbox[3][0], bbox[3][1], bbox[2][0], bbox[2][1]]
                rects.append(rect)
                labels.append(d["content"])
        img = cv2.imread(img_path.lstrip("/"))
        pred_texts = recognizer.infer(img, rects, labels)
        item = {}
        boxes = []
        item["img"] = img_path.split("/")[-1]
        for pred, label, rect in zip(pred_texts, labels, rects):
            box = {}
            box["predict"] = pred
            box["label"] = label
            box["bbox"] = rect
            boxes.append(box)
        item["boxes"] = boxes
        result.append(item)
    with open("result.json", "w", encoding='utf-8') as f:
        f.write(json.dumps(result, ensure_ascii=False))

    with open("result.txt", "a") as f:
        for f_name in os.listdir(args.image_folder):
            print(f_name)
            if f_name.endswith("jpg") or f_name.endswith("jpeg") or f_name.endswith("png") or f_name.endswith("JPG"):
                img = Image.open(args.image_folder + "/" + f_name).convert('L')
                print(img)
                text = recognizer.infer(img)
                f.write(f_name + "\t" + text + "\n")
