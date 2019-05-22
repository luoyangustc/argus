#include "pre_process.hpp"
#include <opencv2/opencv.hpp>
#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace tron {
namespace terror_mixup {
using std::string;
using std::vector;

TEST(TerrorMixup, DetPreProcessImage) {
  cv::Mat im_ori =
      cv::imread("../src/app/terror_mixup/testdata/1.png", cv::IMREAD_COLOR);
  /*
import cv2,numpy as np
im=cv2.resize(cv2.imread('../src/app/terror_mixup/testdata/1.png'),(5,6))
im = im.astype(np.float32)
im = im - np.array([[[103.52, 116.28, 123.675]]])
im = im * 0.017
im = im.transpose((2, 0, 1))
im = im.reshape(-1)
print(','.join(['%.6f'%i for i in im]))
  */
  const float expect[] = {
      0.824160, 2.371160, 2.320160, -0.314840, 0.722160, 2.405160,
      2.439160, 2.507160, 1.317160, 0.671160, 2.456160, 2.439160,
      2.337160, 1.147160, 0.297160, 1.623160, 2.286160, 2.388160,
      -0.654840, -0.960840, 2.218160, 2.473160, 2.473160, 2.456160,
      2.456160, 2.507160, 2.507160, 2.507160, 2.507160, 2.507160,
      -0.004760, 2.205240, 1.661240, -1.075760, 0.522240, 2.290240,
      2.341240, 2.341240, 1.321240, 0.454240, 2.341240, 2.222240,
      2.239240, 1.219240, -0.072760, 1.559240, 2.205240, 2.290240,
      -0.497760, -1.109760, 2.103240, 2.358240, 2.358240, 2.358240,
      2.324240, 2.290240, 2.290240, 2.290240, 2.290240, 2.290240,
      1.229525, 2.113525, 2.181525, -1.184475, 1.008525, 2.181525,
      2.215525, 2.198525, 1.450525, 1.076525, 2.147525, 2.198525,
      2.164525, 1.484525, 0.634525, 1.603525, 2.130525, 2.164525,
      2.232525, -0.232475, 1.960525, 2.215525, 2.215525, 2.198525,
      2.215525, 2.164525, 2.164525, 2.164525, 2.164525, 2.164525};
  EXPECT_THAT(expect, testing::Pointwise(testing::FloatNear(1e-6),
                                         det_pre_process_image(im_ori, 6, 5)));
}

TEST(TerrorMixup, ClsPreProcessImage) {
  cv::Mat im_ori =
      cv::imread("../src/app/terror_mixup/testdata/1.png", cv::IMREAD_COLOR);
  /*
import cv2,numpy as np


def center_crop(img, crop_size):
    short_edge = min(img.shape[:2])
    if short_edge < crop_size:
        raise ErrorBase(400, "bad image size")
    yy = int((img.shape[0] - crop_size) / 2)
    xx = int((img.shape[1] - crop_size) / 2)
    return img[yy:yy + crop_size, xx:xx + crop_size]


def cls_preProcessImage(img=None):
    img = img.astype(np.float32)
    img = cv2.resize(img, (10, 12))

    img = center_crop(img, 8)
    img -= np.array([[[103.94, 116.78, 123.68]]])
    img = img * 0.017
    img = img.transpose((2, 0, 1))
    return img
img = cv2.imread('../src/app/terror_mixup/testdata/1.png')
img = cls_preProcessImage(img)
img = img.reshape(-1)
print(','.join(['%.6f'%i for i in img]))
  */
  const float expect[] = {
      2.449020, 2.461770, 2.432020, 2.421820, 2.061420, 0.777920,
      -0.789480, 1.465570, -0.503030, 2.338520, 1.254770, 2.480470,
      0.049470, -0.312630, -0.207230, 0.720120, 1.949220, 2.534020,
      -1.203430, 2.272220, -0.863430, -0.754630, 0.315520, 1.591370,
      -0.853230, -1.269730, 2.359770, 2.508520, 0.331670, 1.786870,
      -1.065730, 1.330420, 1.960270, 0.842520, 2.346170, 2.299420,
      -0.984980, 2.460070, -1.069980, 0.769420, 2.416720, 2.508520,
      2.484720, 2.425220, 2.456670, 2.466020, -0.984980, 0.160820,
      2.466020, 2.466020, 2.484720, 2.466020, 2.466020, 2.466020,
      -0.729980, 0.911370, 2.415020, 2.415020, 2.431170, 2.415020,
      2.415020, 2.415020, 2.079270, 0.800020, 2.298740, 2.324240,
      2.316590, 2.332740, 1.852490, 0.921740, -1.428510, 1.359490,
      0.461040, 2.192490, 1.511640, 2.342090, 0.780640, -0.224910,
      -0.693260, 0.514590, 1.873740, 2.349740, -0.281010, 2.132990,
      -0.162860, -0.419560, -0.034510, 1.550740, -0.559810, -0.667760,
      2.250290, 2.259640, 0.418540, 1.747090, -1.615510, 1.233690,
      1.858440, 0.658240, 2.285990, 2.128740, -1.059610, 2.349740,
      -1.598510, 0.661640, 2.334440, 2.281740, 2.331040, 2.349740,
      2.306390, 2.349740, -1.581510, 0.052190, 2.349740, 2.349740,
      2.314040, 2.349740, 2.349740, 2.349740, -0.480760, 1.009290,
      2.247740, 2.247740, 2.238390, 2.247740, 2.247740, 2.247740,
      1.954490, 0.820590, 2.206940, 2.202690, 2.128740, 2.175490,
      2.042890, 1.739440, -1.516060, 1.827840, 1.891590, 2.185690,
      2.059040, 2.127890, 1.932390, 1.822740, -0.139060, 1.289790,
      2.173790, 2.092190, 1.730940, 2.116840, 1.847390, 2.083690,
      0.647190, 1.944290, 1.791290, 1.905190, 2.141490, 2.188240,
      1.853340, 2.181440, -1.520310, 1.827840, 1.824440, 1.926440,
      2.015690, 2.046290, 1.748790, 2.209490, -1.609560, 1.273640,
      2.200140, 2.215440, 2.200140, 2.215440, 2.223940, 2.215440,
      -1.694560, 0.863940, 2.215440, 2.215440, 2.198440, 2.215440,
      2.215440, 2.215440, 2.079440, 1.828690, 2.121940, 2.121940,
      2.129590, 2.121940, 2.121940, 2.121940, 2.121940, 2.111740};
  EXPECT_THAT(expect,
              testing::Pointwise(testing::FloatNear(1e-4),
                                 cls_pre_process_image(im_ori, 12, 10, 8)));
}

}  // namespace terror_mixup
}  // namespace tron
