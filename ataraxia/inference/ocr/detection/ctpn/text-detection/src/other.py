import cv2
import caffe
import numpy as np
from matplotlib import cm


def prepare_img(im, mean):
    """
        transform img into caffe's input img.
    """
    im_data=np.transpose(im-mean, (2, 0, 1))
    return im_data


def draw_boxes(im, bboxes,f,im_height,im_width,color=None, caption="Image", wait=True):
    """
        boxes: bounding boxes
    """
    text_recs = []

    im = im.copy()
    for box in bboxes:
        single_temp = []
        if color == None:
            if len(box) == 8 or len(box) == 9:
                c = tuple(cm.jet([box[-1]])[0, 2::-1] * 255)
            else:
                c = tuple(np.random.randint(0, 256, 3))
        else:
            c = color
        b1 = box[6] - box[7] / 2
        b2 = box[6] + box[7] / 2
        x1 = box[0]
        y1 = box[5] * box[0] + b1
        x2 = box[2]
        y2 = box[5] * box[2] + b1
        x3 = box[0]
        y3 = box[5] * box[0] + b2
        x4 = box[2]
        y4 = box[5] * box[2] + b2
        disX = x2 - x1
        disY = y2 - y1
        width = np.sqrt(disX * disX + disY * disY)
        fTmp0 = y3 - y1
        fTmp1 = fTmp0 * disY / width
        x = np.fabs(fTmp1 * disX / width)
        y = np.fabs(fTmp1 * disY / width)
        if box[5] < 0:
            x1 -= x
            y1 += y
            x4 += x
            y4 -= y
        else:
            x2 += x
            y2 += y
            x3 -= x
            y3 -= y
        tx1 = int(x1 / f)
        ty1 = int(y1 / f)
        tx2 = int(x2 / f)
        ty2 = int(y2 / f)
        tx3 = int(x4 / f)
        ty3 = int(y4 / f)
        tx4 = int(x3 / f)
        ty4 = int(y3 / f)

        tx1 = (0 if tx1 < 0 else tx1)
        tx1 = (tx1 if tx1 < im_width else (im_width-1))

        ty1 = (0 if ty1<0 else ty1)
        ty1 = (ty1 if ty1 < im_height else (im_height - 1))

        tx2 = (0 if tx2 < 0 else tx2)
        tx2 = (tx2 if tx2 < im_width else (im_width - 1))

        ty2 = (0 if ty2<0 else ty2)
        ty2 = (ty2 if ty2 < im_height else (im_height - 1))

        tx3 = (0 if tx3 < 0 else tx3)
        tx3 = (tx3 if tx3 < im_width else (im_width-1))

        ty3 = (0 if ty3 < 0 else ty3)
        ty3 = (ty3 if ty3 < im_height else (im_height - 1))

        tx4 = (0 if tx4 < 0 else tx4)
        tx4 = (tx4 if tx4 < im_width else (im_width-1))

        ty4 = (0 if ty4 < 0 else ty4)
        ty4 = (ty4 if ty4 < im_height else (im_height - 1))

        single_temp.append(tx1)
        single_temp.append(ty1)
        single_temp.append(tx2)
        single_temp.append(ty2)
        single_temp.append(tx3)
        single_temp.append(ty3)
        single_temp.append(tx4)
        single_temp.append(ty4)
        cv2.line(im, (int(x1), int(y1)), (int(x2), int(y2)), c, 2)
        cv2.line(im, (int(x1), int(y1)), (int(x3), int(y3)), c, 2)
        cv2.line(im, (int(x4), int(y4)), (int(x2), int(y2)), c, 2)
        cv2.line(im, (int(x3), int(y3)), (int(x4), int(y4)), c, 2)
        #cv2.imwrite("sss.jpg",im)
        text_recs.append(single_temp)
    return text_recs



def threshold(coords, min_, max_):
    return np.maximum(np.minimum(coords, max_), min_)


def clip_boxes(boxes, im_shape):
    """
    Clip boxes to image boundaries.
    """
    boxes[:, 0::2]=threshold(boxes[:, 0::2], 0, im_shape[1]-1)
    boxes[:, 1::2]=threshold(boxes[:, 1::2], 0, im_shape[0]-1)
    return boxes

def find_contours(img, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE):
    if cv2.__version__[0] == '2':
        contours2, hierarchy2 = cv2.findContours(img.copy(), mode, method)
    elif cv2.__version__[0] == '3':
        _, contours2, hierarchy2 = cv2.findContours(img.copy(), mode, method)
    return contours2, hierarchy2

def normalize(data):
    if data.shape[0]==0:
        return data
    max_=data.max()
    min_=data.min()
    return (data-min_)/(max_-min_) if max_-min_!=0 else data-min_


def resize_im(im, scale, max_scale=None):
    im_height = im.shape[0]
    im_width = im.shape[1]
    f=float(scale)/min(im.shape[0], im.shape[1])
    if max_scale!=None and f*max(im.shape[0], im.shape[1])>max_scale:
        f=float(max_scale)/max(im.shape[0], im.shape[1])
    return cv2.resize(im, (0, 0), fx=f, fy=f), f,im_height,im_width


class Graph:
    def __init__(self, graph):
        self.graph=graph

    def sub_graphs_connected(self):
        sub_graphs=[]
        for index in xrange(self.graph.shape[0]):
            if not self.graph[:, index].any() and self.graph[index, :].any():
                v=index
                sub_graphs.append([v])
                while self.graph[v, :].any():
                    v=np.where(self.graph[v, :])[0][0]
                    sub_graphs[-1].append(v)
        return sub_graphs


class CaffeModel:
    def __init__(self, net_def_file, model_file):
        self.net_def_file=net_def_file
        self.net=caffe.Net(str(net_def_file), str(model_file), caffe.TEST)

    def blob(self, key):
        return self.net.blobs[key].data.copy()

    def forward(self, input_data):
        return self.forward2({"data": input_data[np.newaxis, :]})

    def forward2(self, input_data):
        for k, v in input_data.items():
            self.net.blobs[k].reshape(*v.shape)
            self.net.blobs[k].data[...]=v
        return self.net.forward()

    def net_def_file(self):
        return self.net_def_file
