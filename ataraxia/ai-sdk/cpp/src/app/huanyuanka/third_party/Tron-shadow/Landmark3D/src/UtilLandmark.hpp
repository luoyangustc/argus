//
//  UtilLandmark.hpp
//  landmark
//
//  Created by xuyixuan on 2018/7/24.
//  Copyright © 2018年 xuyixuan. All rights reserved.
//

#ifndef __UTILLANDMARK_HPP__
#define __UTILLANDMARK_HPP__
#include "NvInfer.h"
#include <stdio.h>
#include <string>
#include <vector>
#include <iostream>
#include <algorithm>
#include "document.h"
#include "stringbuffer.h"
#include "writer.h"
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
using namespace std;
using namespace cv;

namespace Shadow
{
struct Affine_Matrix
{
    Mat affine_mat;
    Mat crop_img;
};
void getBox(string path, string img_name, int resolution, bool &is_find, vector<int> &box);
void getVertices(Mat &pos, int resolution, vector<vector<float>> &all_veritices);
void getLandmark(Mat &pos, vector<vector<float>> &landmark_one);
void matrix2Angle(Mat &p, vector<float> &pose);
void parseRequestBoxes(string &attribute, int resolution, bool &is_find, vector<int> &box);
void getResultJson(vector<vector<float>> &landmark_one, vector<float> &pose, vector<string> &results);
vector<float> estimatePose(vector<vector<float>> &vertices);
Mat P2sRt(Mat p);
rapidjson::Document getDocument(const string &json_text);
} // namespace Shadow
#endif /* UtilLandmark_hpp */
