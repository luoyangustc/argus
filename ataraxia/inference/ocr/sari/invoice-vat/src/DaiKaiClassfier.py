import cv2
import numpy as np
import os
from sklearn import svm
from sklearn.externals import joblib


class clsPredictor_daikai():
    def __init__(self, model_path):
        self.clf = joblib.load(model_path)

    def predict(self, img):
        try:
            mask = img.copy()
            mask[:, 0:int(mask.shape[1] * 3 / 7)] = [0, 0, 0]
            mask = cv2.GaussianBlur(mask, (5, 5), 0)
            mask[np.where(img[:, :, 2] > 170)] = 0
            try:
                color_top = min(np.where(mask[:, :] > 0)[0])
            except:
                color_top = 0

            hist = []
            hist = np.append(hist, np.mean(mask[:, :, 0]))
            hist = np.append(hist, np.mean(mask[:, :, 1]))
            hist = np.append(hist, np.mean(mask[:, :, 2]))
            hist = np.append(hist, color_top)

            predict = self.clf.predict([hist])
        except:
            predict = None
        return predict
