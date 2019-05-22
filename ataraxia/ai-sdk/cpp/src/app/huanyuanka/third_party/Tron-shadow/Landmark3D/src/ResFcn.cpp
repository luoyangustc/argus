//
//  ResFcn.cpp
//  landmark
//
//  Created by xuyixuan on 2018/7/24.
//  Copyright © 2018年 xuyixuan. All rights reserved.
//

#include "ResFcn.hpp"
#include "UtilLandmark.hpp"
#include "Util.hpp"
#include "document.h"
#include "stringbuffer.h"
#include "writer.h"
#include "NvInfer.h"
#include "Data.hpp"
#include <vector>
#include <dirent.h>
#include <unistd.h>
#include <chrono>
#include <numeric>
#include <memory>
#include <fstream>
#include <sstream>
#include <string>
#include <time.h>
#include <cstring>
#include <sys/stat.h>
#include <cuda_runtime_api.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace nvuffparser;
using namespace nvinfer1;
using namespace std;
using namespace cv;
using namespace rapidjson;

namespace Shadow
{
static Logger gLogger;

resFcn::resFcn(int batchSize, const int *inputShape, float *preParam, InterMethod interMethod)
{
    this->BATCH_SIZE = batchSize;
    this->INPUT_CHANNELS = inputShape[0];
    this->INPUT_WIDTH = inputShape[1];
    this->INPUT_HEIGHT = inputShape[2];
}

ShadowStatus resFcn::init(const int gpuID, void *data, const int size)
{
    INPUT_BLOB_NAME = "Placeholder";
    OUTPUT_BLOB_NAME = "resfcn256/Conv2d_transpose_16/Sigmoid";
    try
    {
        cudaSetDevice(gpuID);
    }
    catch (...)
    {
        return shadow_status_set_gpu_error;
    }

    try
    {
        runtime = createInferRuntime(gLogger);
        engine = runtime->deserializeCudaEngine(data, size, nullptr);
        context = engine->createExecutionContext();
    }
    catch (...)
    {
        return shadow_status_deserialize_error;
    }
    try
    {
        input_index = engine->getBindingIndex(INPUT_BLOB_NAME.c_str());
        output_index = engine->getBindingIndex(OUTPUT_BLOB_NAME.c_str());
    }
    catch (...)
    {
        return shadow_status_blobname_error;
    }
    try
    {
        cudaMalloc(&buffers[input_index], BATCH_SIZE * INPUT_WIDTH * INPUT_HEIGHT * INPUT_CHANNELS * sizeof(float));
        cudaMalloc(&buffers[output_index], BATCH_SIZE * INPUT_WIDTH * INPUT_HEIGHT * INPUT_CHANNELS * sizeof(float));
    }
    catch (...)
    {
        return shadow_status_cuda_malloc_error;
    }

    run_num = 1;
    iteration = 1;
    resolution = INPUT_WIDTH;
    return shadow_status_success;
}

vector<float> resFcn::preProcess(const vector<Mat> &imgs, vector<string> attribute, vector<Mat> &affine_matrix)
{
    vector<int> box;
    Mat img;
    Affine_Matrix tmp_affine_mat;
    vector<float> data;

    for (int i = 0; i < BATCH_SIZE; ++i)
    {
        img = imgs[i];
        Mat similar_img(resolution, resolution, img.type());
        bool is_find = false;

        parseRequestBoxes(attribute[i], resolution, is_find, box);

        int old_size = (box[1] - box[0] + box[3] - box[2]) / 2;
        int size = old_size * 1.58;
        float center_x = 0.0, center_y = 0.0;
        box[3] = box[3] - old_size * 0.3;
        box[1] = box[1] - old_size * 0.25;
        box[0] = box[0] + old_size * 0.25;
        center_x = box[1] - (box[1] - box[0]) / 2.0;
        center_y = box[3] - (box[3] - box[2]) / 2.0 + old_size * 0.14;

        float temp_src[3][2] = {{center_x - size / 2, center_y - size / 2}, {center_x - size / 2, center_y + size / 2}, {center_x + size / 2, center_y - size / 2}};

        Mat srcMat(3, 2, CV_32F, temp_src);
        float temp_dest[3][2] = {{0, 0}, {0, static_cast<float>(resolution - 1)}, {static_cast<float>(resolution - 1), 0}};
        Mat destMat(3, 2, CV_32F, temp_dest);
        Mat affine_mat = getAffineTransform(srcMat, destMat);

        img.convertTo(img, CV_32FC3);

        img = img / 255.;

        warpAffine(img, similar_img, affine_mat, similar_img.size());

        affine_matrix.push_back(affine_mat); //

        for (int c = 0; c < INPUT_CHANNELS; ++c)
        {
            for (int row = 0; row < INPUT_WIDTH; row++)
            {
                for (int col = 0; col < INPUT_HEIGHT; col++)
                {
                    data.push_back(similar_img.at<Vec3f>(row, col)[c]);
                }
            }
        }
    }
    return data;
}

ShadowStatus resFcn::predict(const vector<Mat> &imgs, const vector<string> &attributes, vector<string> &results)
{
    Mat img;
    vector<Mat> network_out_all;
    vector<Mat> position_map;
    vector<Mat> affine_matrix;
    float *out_data = NULL;
    ShadowStatus status;
    vector<float> network_out(BATCH_SIZE * INPUT_CHANNELS * INPUT_HEIGHT * INPUT_WIDTH);
    vector<float> data = preProcess(imgs, attributes, affine_matrix);

    status = doInference(&data[0], &network_out[0], BATCH_SIZE);
    if (status != shadow_status_success)
    {
        return status;
    }
    for (int i = 0; i < BATCH_SIZE; ++i)
    {
        vector<float> my_data;
        my_data.clear();
        Mat network_out_img(INPUT_WIDTH, INPUT_HEIGHT, CV_32FC3);
        out_data = &network_out[0] + i * INPUT_WIDTH * INPUT_HEIGHT * INPUT_CHANNELS;

        for (int j = 0; j < INPUT_WIDTH * INPUT_HEIGHT * INPUT_CHANNELS; ++j)
        {
            my_data.push_back(out_data[j] * INPUT_WIDTH * 1.1);
        }
        int n = 0;
        for (int row = 0; row < INPUT_WIDTH; row++)
        {
            for (int col = 0; col < INPUT_HEIGHT; col++)
            {
                network_out_img.at<Vec3f>(row, col)[2] = my_data[n];
                ++n;
                network_out_img.at<Vec3f>(row, col)[1] = my_data[n];
                ++n;
                network_out_img.at<Vec3f>(row, col)[0] = my_data[n];
                ++n;
            }
        }
        network_out_all.push_back(network_out_img);
    }
    out_data = NULL;
    position_map = postProcess(affine_matrix, network_out_all);
    if (position_map.empty())
    {
        return shadow_status_data_error;
    }
    dealResult(results, position_map);
    return shadow_status_success;
}

vector<Mat> resFcn::postProcess(vector<Mat> &affine_matrix, vector<Mat> &network_out)
{
    vector<Mat> position_map;
    vector<float> face_ind;

    Mat img, z, vertices_T, stacked_vertices, affine_mat_stack;
    Mat pos(resolution, resolution, CV_8UC3);
    string name;
    for (int i = 0; i < BATCH_SIZE; ++i)
    {
        string tmp = "";
        Mat affine_mat, affine_mat_inv;

        img = network_out[i];
        affine_mat = affine_matrix[i];
        invertAffineTransform(affine_mat, affine_mat_inv);

        Mat cropped_vertices(resolution * resolution, 3, img.type()),
            cropped_vertices_T(3, resolution * resolution, img.type());

        cropped_vertices = img.reshape(1, resolution * resolution);
        Mat cropped_vertices_swap(resolution * resolution, 3, cropped_vertices.type());

        cropped_vertices.col(0).copyTo(cropped_vertices_swap.col(2));
        cropped_vertices.col(1).copyTo(cropped_vertices_swap.col(1));
        cropped_vertices.col(2).copyTo(cropped_vertices_swap.col(0));

        transpose(cropped_vertices_swap, cropped_vertices_T);
        cropped_vertices_T.convertTo(cropped_vertices_T, affine_mat.type());
        z = cropped_vertices_T.row(2).clone() / affine_mat.at<double>(0, 0);

        Mat ones_mat(1, resolution * resolution, cropped_vertices_T.type(), Scalar(1));
        ones_mat.copyTo(cropped_vertices_T.row(2));

        cropped_vertices_T.convertTo(cropped_vertices_T, affine_mat.type());

        Mat vertices = affine_mat_inv * cropped_vertices_T;
        z.convertTo(z, vertices.type());

        vconcat(vertices.rowRange(0, 2), z, stacked_vertices);
        transpose(stacked_vertices, vertices_T);
        pos = vertices_T.reshape(3, resolution);
        Mat pos2(resolution, resolution, CV_64FC3);

        for (int row = 0; row < pos.rows; ++row)
        {
            for (int col = 0; col < pos.cols; ++col)
            {
                pos2.at<Vec3d>(row, col)[0] = pos.at<Vec3d>(row, col)[2];
                pos2.at<Vec3d>(row, col)[1] = pos.at<Vec3d>(row, col)[1];
                pos2.at<Vec3d>(row, col)[2] = pos.at<Vec3d>(row, col)[0];
            }
        }
        position_map.push_back(pos2);
    }
    return position_map;
}

//deal with position map
void resFcn::dealResult(vector<string> &results, vector<Mat> &position_map)
{
    for (int i = 0; i < BATCH_SIZE; i++)
    {
        vector<vector<float>> all_vertices(face_ind.size(), vector<float>(3, 0));
        getVertices(position_map[i], resolution, all_vertices);

        vector<vector<float>> landmark_one(68, vector<float>(3, 0));
        getLandmark(position_map[i], landmark_one);

        vector<float> pose(3, 0);
        pose = estimatePose(all_vertices);

        getResultJson(landmark_one, pose, results);
    }
}

ShadowStatus resFcn::doInference(float *input_data, float *output_data, int batch_size)
{
    float *tmp_data = NULL;
    size_t mem_size = INPUT_WIDTH * INPUT_HEIGHT * INPUT_CHANNELS * sizeof(float);

    for (int i = 0; i < iteration; i++)
    {
        float total = 0, ms;
        for (int run = 0; run < run_num; run++)
        {
            //*create space for input and set the input data*/
            tmp_data = &input_data[0] + run * INPUT_WIDTH * INPUT_HEIGHT * INPUT_CHANNELS;
            try
            {
                cudaMemcpyAsync(buffers[input_index], tmp_data, 1 * INPUT_CHANNELS * INPUT_WIDTH * INPUT_HEIGHT * sizeof(float), cudaMemcpyHostToDevice);
            }
            catch (...)
            {
                return shadow_status_cuda_memcpy_error;
            }

            context->execute(batch_size, &buffers[0]);
            /*
                auto t_start = chrono::high_resolution_clock::now();
                auto t_end = chrono::high_resolution_clock::now();
                ms = chrono::duration<float, milli>(t_end - t_start).count();
                total += ms;
                total /= batch_size;
                cout << "Average over " << run_num << " runs is " << total << " ms." << endl;
                */
            tmp_data = &output_data[0] + run * INPUT_WIDTH * INPUT_HEIGHT * INPUT_CHANNELS;
            try
            {
                cudaMemcpyAsync(tmp_data, buffers[output_index], mem_size, cudaMemcpyDeviceToHost);
            }
            catch (...)
            {
                return shadow_status_cuda_memcpy_error;
            }
        }
    }
    tmp_data = NULL;
    return shadow_status_success;
}

ShadowStatus resFcn::destroy()
{
    try
    {
        cudaFree(buffers[output_index]);
        cudaFree(buffers[input_index]);
    }
    catch (...)
    {
        return shadow_status_cuda_free_error;
    }
    context->destroy();
    engine->destroy();
    runtime->destroy();
    delete this;
    return shadow_status_success;
}
} // namespace Shadow
