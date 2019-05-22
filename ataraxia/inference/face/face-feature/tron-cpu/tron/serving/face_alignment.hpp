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

const int FEA_LENGTH=512;

float getMold(float* vec){   //求向量的模长
        float sum = 0.0;
        for (int i = 0; i<FEA_LENGTH; ++i)
            sum += vec[i] * vec[i];
        return sqrt(sum);
    }

float getSimilarity( float* lhs,  float* rhs){
    
        float tmp = 0.0;  //内积
        for (int i = 0; i<FEA_LENGTH; ++i)
            tmp += lhs[i] * rhs[i];
        return tmp / (getMold(lhs)*getMold(rhs));
    }


// non reflective transform
cv::Mat findNonReflectiveTransform(const std::vector<cv::Point2d>& source_points, 
                                   std::vector<cv::Point2d>& target_points, 
                                   Mat& Tinv /* = Mat()*/) {
    
    assert(source_points.size() == target_points.size());
    assert(source_points.size() >= 2);

    /*
    u = uv(:,1); // first col
    v = uv(:,2); // second col
    U = [u; v];
    */
    Mat U = Mat::zeros(target_points.size() * 2, 1, CV_64F); // 转为列向量
    Mat X = Mat::zeros(source_points.size() * 2, 4, CV_64F);

    for (int i = 0; i < target_points.size(); i++) {

        U.at<double>(i * 2, 0) = source_points[i].x;
        U.at<double>(i * 2 + 1, 0) = source_points[i].y;

        /*
        X = [x   y  ones(M,1)   zeros(M,1);
            y  -x  zeros(M,1)  ones(M,1)  ];
        */
        X.at<double>(i * 2, 0) = target_points[i].x; 
        X.at<double>(i * 2, 1) = target_points[i].y;
        X.at<double>(i * 2, 2) = 1;
        X.at<double>(i * 2, 3) = 0;

        X.at<double>(i * 2 + 1, 0) = target_points[i].y;
        X.at<double>(i * 2 + 1, 1) = -target_points[i].x;
        X.at<double>(i * 2 + 1, 2) = 0;
        X.at<double>(i * 2 + 1, 3) = 1;

    }
    //  cout<<"\nU:"<<U<<"\n";
    //  cout<<"\nX:"<<X<<"\n";

    /*
    sc = r(1);
    ss = r(2);
    tx = r(3);
    ty = r(4);

    Tinv = [sc -ss 0;
            ss  sc 0;
            tx  ty 1];

    T = inv(Tinv);

    */
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
/*   for(int i=0;i<5;++i){
 cout<<"target.x="<<target_points[i].x<<",target.y="<<target_points[i].y<<"\n";
  }
*/
 /*  for(int i=0;i<5;++i){
 cout<<"\nsource.x="<<source_points[i].x<<",source.y="<<source_points[i].y<<"\n";
    } 
*/
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
  /*  std::cout<<"trans1:\n"<<trans1<<"\n";
    std::cout<<"trans2:\n"<<trans2<<"\n";
    std::cout<<"trans_points1:\n"<<trans_points1<<"\n";
    std::cout<<"trans_points2:\n"<<trans_points2<<"\n";
    std::cout<<"before norm...\n";*/
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
人脸对齐，输入原图，5点坐标，输出对齐后图像
*/

void faceAlignmet(
          const cv::Mat& src,std::vector<std::vector<Point2d>>& fivePts,
          std::vector<cv::Mat>& algnedFaces,
          const int  face_size_type=FACE_SIZE_TYPE::FACE_96_112){
  
 //  cout<<"test0\n";
    if(!src.data){
        // std::cout<<"输入图像数据有误\n";
        std::cout << "Wrong input image data!" << std::endl;
        return;
    }
    if(fivePts.empty()){
        // std::cout<<"关键点数据为空\n";
        std::cout << "empty input landmarks!" << std::endl;
        return;
    }
    // cout<<"test1\n";
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

 /*
const int size_test=fivePts.size();
  cout<<"size_test="<<size_test<<"\n";
  for(auto& iter: fivePts){
  for(int i=0;i<5;++i)
 cout<<"fivePts.x="<<iter[i].x<<",fivePts.y="<<iter[i].y<<"\n";
}
*/
   for(auto& iter: fivePts){
   
    Mat transform,Tinv=Mat();
    transform =findSimilarityTransform(iter,target_points,Tinv,face_size_type);
    
   //cout<<"after transform....\n";
  // cout<<"\n transform=\n"<<transform<<"\n";

   // Mat transformT;
   // transpose(transform,transformT);
   // cout<<"\n transformT=\n"<<transformT<<"\n";

   Mat cropImage; 
   Size sz;
   ++num;
   stringstream ss;
   string filename;
   ss<<num;
   ss>>filename;
   filename+=".png";
  // std::cout<<"filename: "<<filename<<".\n";
   switch(face_size_type){  
     case 0: sz=Size(96, 112);  break;
     case 1: sz=Size(112, 112); break;
     default:sz=Size(0,0);      break;

   } // switch

   warpAffine(src, cropImage, transform,sz);
   algnedFaces.push_back(cropImage);
   
//std::cout<<"width="<<cropImage.cols<<",height="<<cropImage.rows<<"\n";
// cv::imwrite("test-crop2.png",cropImage);
//  cv::waitKey();
//  std::cout<<"save an image done:"<<filename<<"\n";
   
   /*
   imshow("test",cropImage);
   int key=waitKey();
   if(key==27)
   break;
   */

   } // for

   return;

}


#endif 
