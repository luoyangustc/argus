#ifndef _FACE_ALIGNMENT_H_
#define _FACE_ALIGNMENT_H_

#include "opencv2/opencv.hpp"
#include<iostream>
#include<string>

using namespace std;
using namespace cv;

enum FACE_SIZE_TYPE{
 FACE_96_112=0,
 FACE_112_112=1
};

// non reflective transform
cv::Mat findNonReflectiveTransform(const std::vector<cv::Point2d>& source_points, 
                                   std::vector<cv::Point2d>& target_points, 
                                   Mat& Tinv /* = Mat()*/) {
    
    assert(source_points.size() == target_points.size());
    assert(source_points.size() >= 2);

    Mat U = Mat::zeros(target_points.size() * 2, 1, CV_64F); // 转为列向量
    Mat X = Mat::zeros(source_points.size() * 2, 4, CV_64F);

    for (int i = 0; i < target_points.size(); i++) {

        U.at<double>(i * 2, 0) = source_points[i].x;
        U.at<double>(i * 2 + 1, 0) = source_points[i].y;

        X.at<double>(i * 2, 0) = target_points[i].x; 
        X.at<double>(i * 2, 1) = target_points[i].y;
        X.at<double>(i * 2, 2) = 1;
        X.at<double>(i * 2, 3) = 0;

        X.at<double>(i * 2 + 1, 0) = target_points[i].y;
        X.at<double>(i * 2 + 1, 1) = -target_points[i].x;
        X.at<double>(i * 2 + 1, 2) = 0;
        X.at<double>(i * 2 + 1, 3) = 1;

    }

    Mat r = X.inv(DECOMP_SVD)*U;
    Tinv = (Mat_<double>(3, 3) << r.at<double>(0), -r.at<double>(1), 0,
                         r.at<double>(1), r.at<double>(0), 0,
                         r.at<double>(2), r.at<double>(3), 1);
    Mat T = Tinv.inv(DECOMP_SVD);
    Tinv = Tinv(Rect(0, 0, 2, 3)).t();
    return T(Rect(0,0,2,3)).t();
  }


//method 2
cv::Mat findSimilarityTransform(const std::vector<cv::Point2d>& source_points,
                                std::vector<cv::Point2d>& target_points, 
                                Mat& Tinv /*= Mat()*/,const int  face_size_type ) {
    Mat Tinv1,Tinv2;
    Mat trans1 = findNonReflectiveTransform(source_points, target_points, Tinv1);
    std::vector<cv::Point2d> source_point_reflect;

    for (auto sp : source_points) {
      source_point_reflect.push_back(Point2d(-sp.x, sp.y));
    }

   cv::Mat trans2 = findNonReflectiveTransform(source_point_reflect, target_points, Tinv2);
    trans2.colRange(0,1) *= -1;
    std::vector<cv::Point2d> trans_points1, trans_points2;
    transform(source_points, trans_points1, trans1);
    transform(source_points, trans_points2, trans2);

    source_point_reflect.clear();
    double norm1 = norm(Mat(trans_points1), Mat(target_points), NORM_L2);
    double norm2 = norm(Mat(trans_points2), Mat(target_points), NORM_L2);
    Tinv = norm1 < norm2 ? Tinv1 : Tinv2;
 //  std::cout<<"after norm...\n";
   trans_points1.clear();
   trans_points2.clear();
    return norm1 < norm2 ? trans1 : trans2;
  }



 /* 
人脸对齐，输入检测到的人脸图，5点坐标，输出对齐后图像
*/

void faceAlignmet(
          const cv::Mat& src,std::vector<Point2d>& fivePts,
          std::vector<cv::Mat>& algnedFaces,
          const int  face_size_type=FACE_SIZE_TYPE::FACE_112_112){
  
    if(!src.data){
        std::cout << "Wrong input image data!" << std::endl;
        return;
    }
    if(fivePts.empty()){
        std::cout << "empty input landmarks!" << std::endl;
        return;
    }
    std::vector<cv::Point2d> source_points,target_points;
    static int num=0;
    if(face_size_type==FACE_SIZE_TYPE::FACE_96_112){
    target_points.push_back(Point2d(30.29459953,  51.69630051));
    target_points.push_back(Point2d(65.53179932,  51.50139999));
    target_points.push_back(Point2d(48.02519989,  71.73660278));
    target_points.push_back(Point2d(33.54930115,  92.3655014));
    target_points.push_back(Point2d(62.72990036,  92.20410156));

    }

    else if(face_size_type==FACE_SIZE_TYPE::FACE_112_112){
    // left and right size added 8 pixels,respectively.
    target_points.push_back(Point2d(30.29459953+8.0,  51.69630051));
    target_points.push_back(Point2d(65.53179932+8.0,  51.50139999));
    target_points.push_back(Point2d(48.02519989+8.0,  71.73660278));
    target_points.push_back(Point2d(33.54930115+8.0,  92.3655014));
    target_points.push_back(Point2d(62.72990036+8.0,  92.20410156));
}

  //  for(auto& iter: fivePts){
   
  //     Mat transform,Tinv=Mat();
  //     transform =findSimilarityTransform(iter,target_points,Tinv,face_size_type);

  //    Mat cropImage; 
  //    Size sz;
  // //    ++num;
  // //    stringstream ss;
  // //    string filename;
  // //    ss<<num;
  // //    ss>>filename;
  // //    filename+=".png";
  //    switch(face_size_type){  
  //      case 0: sz=Size(96, 112);  break;
  //      case 1: sz=Size(112, 112); break;
  //      default:sz=Size(0,0);      break;

  //    } // switch

  //    warpAffine(src, cropImage, transform,sz);


  //   // cv::Vec3b elem_3 = cropImage.at<cv::Vec3b>(0,0);
  //   //   printf("before %d %d %d\n", elem_3.val[0], elem_3.val[1], elem_3.val[2]);
  //   //      cv::imwrite("result/cropimage.jpg",cropImage);

  //    cv::cvtColor(cropImage, cropImage, cv::COLOR_BGR2RGB);
  //    algnedFaces.push_back(cropImage);
  //   //  for(int i = 0;i < 112;i++)
  //   //   for(int j = 0;j<112;j++){
  //   //     elem_3 = cropImage.at<cv::Vec3b>(i,j);
  //   //   printf(" %d %d %d %d %d\n",i,j, elem_3.val[0], elem_3.val[1], elem_3.val[2]);
  //   // }
  // //  std::cout<<"save an image done:"<<filename<<"\n";
     
  //    /*
  //    imshow("test",cropImage);
  //    int key=waitKey();
  //    if(key==27)
  //    break;
  //    */

  //  } // for

   
  Mat transform,Tinv=Mat();
  transform =findSimilarityTransform(fivePts,target_points,Tinv,face_size_type);

  Mat cropImage; 
  Size sz;
  switch(face_size_type){  
  case 0: sz=Size(96, 112);  break;
  case 1: sz=Size(112, 112); break;
  default:sz=Size(0,0);      break;
  } // switch

  warpAffine(src, cropImage, transform, sz);

  cv::cvtColor(cropImage, cropImage, cv::COLOR_BGR2RGB);
  algnedFaces.push_back(cropImage);


  return;

}


#endif 
