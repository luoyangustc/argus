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


def draw_boxes(im, bboxes, is_display=True, color=None, caption="Image", wait=True):
    """
        boxes: bounding boxes
    """
    if len(bboxes) ==0:
        return None

    im=im.copy()
    for box in bboxes:
        if color==None:
            if len(box)==5 or len(box)==9:
                c=tuple(cm.jet([box[-1]])[0, 2::-1]*255)
            else:
                c=tuple(np.random.randint(0, 256, 3))
        else:
            c=color
        cv2.rectangle(im, tuple(box[:2]), tuple(box[2:4]), c)
    if is_display:
        cv2.imshow(caption, im)
        if wait:
            cv2.waitKey(0)
    return im


def threshold(coords, min_, max_):
    return np.maximum(np.minimum(coords, max_), min_)


def clip_boxes(boxes, im_shape):
    """
    Clip boxes to image boundaries.
    """
    boxes[:, 0::2]=threshold(boxes[:, 0::2], 0, im_shape[1]-1)
    boxes[:, 1::2]=threshold(boxes[:, 1::2], 0, im_shape[0]-1)
    return boxes

def refine_boxes(im, boxes, expand_pixel_len = 10, pixel_blank = 2, binary_thresh=150):
    im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    ret, im_binary = cv2.threshold(im_gray, binary_thresh, 255, cv2.THRESH_BINARY_INV)
    boxes_new = []
    for bbox in boxes:
        x1 = int(np.floor(bbox[0]))
        y1 = int(np.floor(bbox[1]))
        x2 = int(np.ceil(bbox[2]))
        y2 = int(np.ceil(bbox[3]))
        x1_ex = max(x1 - expand_pixel_len, 0)
        x2_ex = min(x2 + expand_pixel_len, im_binary.shape[0] - 1)
        im_chip = im_binary[y1:y2, x1_ex:x2_ex]
        if im_chip.size == 0:
            continue
        if im_chip[0,-1] == 255:  # remove useless texts (with black background)
            continue

        contours, hierarchy = find_contours(im_chip)
        if len(contours) == 0:
            continue #boxes_new.append([x1,y1,x2,y2])

        xmin = 100000000000
        xmax = 0
        for cnt in contours:
            x,y,w,h = cv2.boundingRect(cnt)
            if x < xmin:
                xmin = x
            if x+w > xmax:
                xmax = x+w
        x1_new = max(x1_ex+xmin-pixel_blank, 0)
        x2_new = min(x1_ex+xmax+pixel_blank,im_binary.shape[0]-1)
        boxes_new.append([x1_new,y1,x2_new,y2])
    return boxes_new

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
    f=float(scale)/min(im.shape[0], im.shape[1])
    if max_scale!=None and f*max(im.shape[0], im.shape[1])>max_scale:
        f=float(max_scale)/max(im.shape[0], im.shape[1])
    return cv2.resize(im, (0, 0), fx=f, fy=f), f

def calc_area_ratio(text_lines, img_shape):
    im_height, im_width, _ = img_shape
    im_area = im_height * im_width
    text_area = 0
    for rect in text_lines:
        left = rect[0]
        top = rect[1]
        right = rect[2]
        bottom = rect[3]
        rect_area = (right - left) * (bottom - top)
        text_area += rect_area

    text_area_ratio = float(text_area) / im_area
    return text_area_ratio

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
