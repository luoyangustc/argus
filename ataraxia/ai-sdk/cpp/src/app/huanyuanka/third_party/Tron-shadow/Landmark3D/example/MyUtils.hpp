//
//  MyUtils.hpp
//  landmark
//
//  Created by xuyixuan on 2018/7/24.
//  Copyright © 2018年 xuyixuan. All rights reserved.
//
#ifndef __MYUTILS_HPP__
#define __MYUTILS_HPP__
#include "../src/UtilLandmark.hpp"
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
#include <document.h>
#include <stringbuffer.h>
#include <writer.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace std;
using namespace cv;

namespace Shadow {
    
    inline ShadowStatus getAllFiles(string path, string suffix, vector<string> &files)
    {
        regex reg_obj(suffix, regex::icase);
        DIR *dp;
        struct dirent *dirp;
        if((dp = opendir(path.c_str())) == NULL)
        {
            return shadow_status_data_error;
        }
        else
        {
            while((dirp = readdir(dp)) != NULL)
            {
                if(dirp->d_type == 8 && regex_match(dirp->d_name, reg_obj))
                {
                    string file_absolute_path = path.c_str();
                    file_absolute_path = file_absolute_path.append("/");
                    file_absolute_path = file_absolute_path.append(dirp->d_name);
                    files.push_back(file_absolute_path);
                }
            }
        }
        closedir(dp);
        return shadow_status_success;
    }

    inline bool searchKey(vector<int> a, int value)
    {
        for(uint i=0;i<a.size();i++)
        {
            if(a[i]==value)
                return true;
        }
        return false;
    }
    
    inline void plotLandmark(Mat &img, string name, vector<vector<float>> &kpt, string plot_path)
    {
        Mat image = img.clone();
        vector<int> end_list = {17-1, 22-1, 27-1, 42-1, 48-1, 31-1, 36-1, 68-1};
        for(uint i = 0; i < 68; i++)
        {
            int start_point_x, start_point_y, end_point_x, end_point_y;
            start_point_x = int(round(kpt[i][0]));
            start_point_y = int(round(kpt[i][1]));
            Point center1(start_point_x,start_point_y);
            circle(image, center1, 2, Scalar(0,0,255));
            
            if (searchKey(end_list,i))
                continue;
            
            end_point_x = int(round(kpt[i+1][0]));
            end_point_y = int(round(kpt[i+1][1]));
            Point center2(end_point_x,end_point_y);
            line(image, center1, center2, Scalar(0,255,0));
        }

        if (access(plot_path.c_str(),6)==-1)
        {
            mkdir(plot_path.c_str(), S_IRWXU);
        }
        imwrite(plot_path + "/" + name, image);
    }

    inline vector<string> mySplit(string my_str,string seperate)
    {
        vector<string> result;
        size_t split_index = my_str.find(seperate);
        size_t start = 0;
        
        while(string::npos!=split_index)
        {
            result.push_back(my_str.substr(start,split_index-start));
            start = split_index+seperate.size();
            split_index = my_str.find(seperate,start);
        }
        result.push_back(my_str.substr(start,split_index-start));
        return result;
    }
    
    inline ShadowStatus parseLandmark(string &attribute, vector<vector<float>> &landmark)
    {

        rapidjson::Document document;
        try
        {
            document = getDocument(attribute);
        }
        catch(...)
        {
            return shadow_status_parse_landmark_error;
        }
        if(document.HasMember("landmark"))
        {
            const auto &landmark_points = document["landmark"];
            if(landmark_points.IsArray())
            {
                for(int i=0;i<68;i++)
                {
                    for(int j=0;j<3;j++)
                        landmark[i][j] = landmark_points[i][j].GetFloat();
                }
            }
            else
            {
                return shadow_status_parse_landmark_error;
            }
        }
        else
        {
            return shadow_status_parse_landmark_error;
        }
        return shadow_status_success;
    }
}
#endif
