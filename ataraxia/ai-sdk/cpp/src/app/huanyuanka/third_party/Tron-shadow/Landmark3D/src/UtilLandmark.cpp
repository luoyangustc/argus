//
//  UtilLandmark.cpp
//  landmark
//
//  Created by xuyixuan on 2018/7/24.
//  Copyright © 2018年 xuyixuan. All rights reserved.
//

#include "UtilLandmark.hpp"
#include "Data.hpp"
#include <iostream>
#include <stdio.h>
#include <cstdlib>
#include <sstream>
#include <string.h>
#include <vector>
#include <fstream>
#include <regex>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <dirent.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace std;
using namespace cv;
using namespace rapidjson;

namespace Shadow
{
rapidjson::Document getDocument(const string &json_text)
{
    rapidjson::Document document;
    if (!json_text.empty())
    {
        document.Parse(json_text.c_str());
    }
    else
    {
        document.Parse("{}");
    }
    return document;
}

inline bool checkValidBoxPts(const float pts[4][2])
{
    if (pts[0][0] == pts[3][0] &&
        pts[0][1] == pts[1][1] &&
        pts[1][0] == pts[2][0] &&
        pts[2][1] == pts[3][1] &&
        pts[2][0] > pts[0][0] &&
        pts[2][1] > pts[0][1])
    {
        return true;
    }
    return false;
}

void parseRequestBoxes(string &attribute, int resolution, bool &is_find, vector<int> &box)
{
    const auto &document = getDocument(attribute);
    /*
        "'attribute' in request data must be a valid json dict string,"
        " and has key 'pts'."
        " pts must be in the form as [[x1,y1],[x2,y1],[x2,y2],[x1,y2]]."
        " all of x1, x2, y1, y2 can be parsed into int values."
        " And also must have (x2>x1 && y2>y1).");
        */

    if (!document.HasMember("detections"))
    {
        is_find = false;
    }

    if (document.HasMember("detections"))
    {
        const auto &ptses = document["detections"];
        if (ptses.Size() == 0)
            is_find = false;
        if (ptses.IsArray())
        {
            const Value &detecition_arr = ptses[0];
            const Value &pts = detecition_arr["pts"];
            if (pts.IsArray())
            {
                float t_pts[4][2];
                try
                {
                    bool isArray = pts.IsArray();
                    if (!isArray)
                    {
                        is_find = false;
                    }
                    const int size = pts.Size();
                    if (size != 4)
                    {
                        is_find = false;
                    }
                    for (int i = 0; i < 4; i++)
                    {
                        for (int j = 0; j < 2; j++)
                        {
                            t_pts[i][j] = pts[i][j].GetInt();
                        }
                    }
                }
                catch (...)
                {
                    is_find = false;
                }
                if (!checkValidBoxPts(t_pts))
                {
                    is_find = false;
                }
                int xmin = t_pts[0][0];
                int ymin = t_pts[0][1];
                int xmax = t_pts[2][0];
                int ymax = t_pts[2][1];

                box.push_back(xmin);
                box.push_back(xmax);
                box.push_back(ymin);
                box.push_back(ymax);
            }
        }
        is_find = true;
    }
}

void getResultJson(vector<vector<float>> &landmark_one, vector<float> &pose, vector<string> &results)
{
    results.clear();
    Document document;
    auto &alloc = document.GetAllocator();
    Value json_result(kObjectType), j_landmark(kArrayType), j_pos(kArrayType);
    for (uint i = 0; i < 68; i++)
    {
        Value points(kArrayType);
        points.PushBack(Value(landmark_one[i][0]), alloc).PushBack(Value(landmark_one[i][1]), alloc).PushBack(Value(landmark_one[i][2]), alloc);
        j_landmark.PushBack(points, alloc);
    }
    json_result.AddMember("landmark", j_landmark, alloc);
    j_pos.PushBack(Value(pose[0]), alloc).PushBack(Value(pose[1]), alloc).PushBack(Value(pose[2]), alloc);
    json_result.AddMember("pose", j_pos, alloc);

    StringBuffer buffer;
    Writer<StringBuffer> writer(buffer);
    json_result.Accept(writer);
    string result = string(buffer.GetString());
    results.push_back(result);
}

void getVertices(Mat &pos, int resolution, vector<vector<float>> &result)
{
    Mat all_vertices = pos.reshape(1, resolution * resolution);
    vector<float>::const_iterator iter;
    int i = 0;
    for (iter = face_ind.begin(); iter != face_ind.end(); iter++)
    {
        result[i][0] = all_vertices.at<double>(int(*iter), 2);
        result[i][1] = all_vertices.at<double>(int(*iter), 1);
        result[i][2] = all_vertices.at<double>(int(*iter), 0);
        i++;
    }
}

void getLandmark(Mat &pos, vector<vector<float>> &landmark_one)
{
    for (uint i = 0; i < uv_kpt_ind1.size(); ++i)
    {
        landmark_one[i][0] = pos.at<Vec3d>(uv_kpt_ind2[i], uv_kpt_ind1[i])[2];
        landmark_one[i][1] = pos.at<Vec3d>(uv_kpt_ind2[i], uv_kpt_ind1[i])[1];
        landmark_one[i][2] = pos.at<Vec3d>(uv_kpt_ind2[i], uv_kpt_ind1[i])[0];
    }
}

vector<float> estimatePose(vector<vector<float>> &vertices)
{
    Mat canonical_vertices_homo;
    Mat canonical_vertices = Mat::zeros(131601 / 3, 3, CV_32FC1);
    vector<float>::const_iterator cv_iter;
    int line = 0;
    for (cv_iter = canonical_vertices_1d.begin(); cv_iter != canonical_vertices_1d.end(); ++line)
    {
        canonical_vertices.at<float>(line, 0) = *cv_iter;
        ++cv_iter;
        canonical_vertices.at<float>(line, 1) = *cv_iter;
        ++cv_iter;
        canonical_vertices.at<float>(line, 2) = *cv_iter;
        ++cv_iter;
    }

    Mat ones_mat(131601 / 3, 1, canonical_vertices.type(), Scalar(1));
    ones_mat.convertTo(ones_mat, CV_32F);
    hconcat(canonical_vertices, ones_mat, canonical_vertices_homo);

    Mat canonical_vertices_homo_T, vertices_T;
    CvMat *canonical_vertices_homo_T_pointer = cvCreateMat(43867, 4, CV_32FC1);
    CvMat *vertices_T_pointer = cvCreateMat(43867, 3, CV_32FC1);
    CvMat *P_pointer = cvCreateMat(4, 3, CV_32FC1);

    cvSetData(canonical_vertices_homo_T_pointer, canonical_vertices_homo.data, CV_AUTOSTEP);

    for (uint i = 0; i < 43867; i++)
    {
        for (uint j = 0; j < 3; j++)
        {
            cvmSet(vertices_T_pointer, i, j, vertices[i][j]);
        }
    }

    cvSolve(canonical_vertices_homo_T_pointer, vertices_T_pointer, P_pointer);

    Mat P(P_pointer->rows, P_pointer->cols, P_pointer->type, P_pointer->data.fl);
    Mat P_T(P_pointer->cols, P_pointer->rows, P_pointer->type);

    transpose(P, P_T);
    vector<float> pose;
    Mat rotation_matrix = P2sRt(P_T);
    matrix2Angle(rotation_matrix, pose);

    cvReleaseMat(&canonical_vertices_homo_T_pointer);
    cvReleaseMat(&vertices_T_pointer);
    cvReleaseMat(&P_pointer);
    return pose;
}

//p 3*4
Mat P2sRt(Mat P)
{
    Mat t2d(2, 1, P.type());
    Mat R1(1, 3, P.type());
    Mat R2(1, 3, P.type());
    Mat R(3, 3, P.type());
    //t2d
    Mat P_row0 = P.rowRange(0, 1).clone();
    R1 = P_row0.colRange(0, 3).clone();
    Mat P_row1 = P.row(1).clone();
    P_row1.colRange(0, 3).copyTo(R2);
    Mat r1 = R1 / norm(R1);
    Mat r2 = R2 / norm(R2);

    CvMat *r1_pointer = cvCreateMat(1, 3, CV_32FC1);
    cvSetData(r1_pointer, r1.data, CV_AUTOSTEP);

    CvMat *r2_pointer = cvCreateMat(1, 3, CV_32FC1);
    cvSetData(r2_pointer, r2.data, CV_AUTOSTEP);

    CvMat *r3_pointer = cvCreateMat(1, 3, CV_32FC1);
    cvCrossProduct(r1_pointer, r2_pointer, r3_pointer);

    Mat r3(r3_pointer->rows, r3_pointer->cols, r3_pointer->type, r3_pointer->data.fl);
    vconcat(r1, r2, R);
    vconcat(R, r3, R);

    cvReleaseMat(&r1_pointer);
    cvReleaseMat(&r2_pointer);
    cvReleaseMat(&r3_pointer);
    return R;
}

//r 3*3
void matrix2Angle(Mat &R, vector<float> &pose_angle)
{
    float x = 0, y = 0, z = 0;
    if (R.at<float>(2, 0) != 1 || R.at<float>(2, 0) != -1)
    {
        x = asin(R.at<float>(2, 0));
        y = atan2(R.at<float>(2, 1) / cos(x), R.at<float>(2, 2) / cos(x));
        z = atan2(R.at<float>(1, 0) / cos(x), R.at<float>(0, 0) / cos(x));
    }
    else
    {
        z = 0;
        if (R.at<float>(2, 0) == -1)
        {
            x = M_PI / 2;
            y = z + atan2(-R.at<float>(0, 1), -R.at<float>(0, 2));
        }
    }
    pose_angle.push_back(x);
    pose_angle.push_back(y);
    pose_angle.push_back(z);
}
} // namespace Shadow
