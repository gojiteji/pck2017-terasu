
#include <iostream>
#include <opencv2/opencv.hpp>
#include "opencv2/core.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/highgui.hpp"
#include "unistd.h"
#include <cmath>
#include <sstream>

#define HEIGHT 1080
#define WIDTH 1920

using namespace cv;
using namespace std;

 cv::VideoCapture cap(0);//デバイスのオープン

void SetVideo(){

 
	   if(!cap.isOpened()){//カメラデバイスが正常にオープンしたか確認．
    cout<<"Could not open the Camera"<<endl;
    exit(1);
   }
   cap.set(cv::CAP_PROP_FRAME_WIDTH,WIDTH);
  cap.set(cv::CAP_PROP_FRAME_HEIGHT,HEIGHT);
	}

Mat PinP_tr(const cv::Mat &srcImg, const cv::Mat &smallImg, const int tx, const int ty)
 {
    //背景画像の作成
    cv::Mat dstImg;
    srcImg.copyTo(dstImg);

    //前景画像の変形行列
    cv::Mat mat = (cv::Mat_<double>(2,3)<<1.0, 0.0, tx, 0.0, 1.0, ty);

    //アフィン変換の実行
    cv::warpAffine(smallImg, dstImg, mat, dstImg.size(), CV_INTER_LINEAR, cv::BORDER_TRANSPARENT);
    return  dstImg;
 }


cv::Mat TakePic(Mat Pic,string windowname="window"){
	
         cap >> Pic;
         cout<<"Pic was saved"<<endl;// break;
return Pic;
}




int main(){
  SetVideo();
//  sleep(100000);
        cv::Mat first(cv::Size(WIDTH,HEIGHT), CV_8UC3);
 
 for(int i=0;i<7;i++)
        first=TakePic(first,"first");
cv::imwrite("/home/pi/Documents/php/Picture/normal.jpg",first);//For being proccessed
cv::Mat srcImg(cv::Size(WIDTH,HEIGHT), CV_8UC3); 
srcImg = cv::imread("/home/pi/Documents/php/Picture/normal.jpg");
   
   srcImg=PinP_tr(srcImg, srcImg, 500, 190);
    cv::Point2f center = cv::Point2f(
        static_cast<float>(srcImg.cols ),
        static_cast<float>(srcImg.rows ));
    double degree = 0.0;  // 回転角度
    double scale = 2.2;   // 拡大率(1.5倍)
//アフィン変換行列
    cv::Mat affine;
    cv::getRotationMatrix2D(center, degree, scale).copyTo(affine);
 
    cv::warpAffine(srcImg, srcImg, affine, srcImg.size(), cv::INTER_CUBIC);
flip(srcImg,srcImg, 1);

Mat small=srcImg;
cv::imwrite("/etc/urao/past.jpg",srcImg);//For sending
transpose(srcImg,srcImg);
cv::imwrite("/home/pi/Documents/php/Picture/normal.jpg",srcImg);//For sending
/*FOR BUILD.PLEASE DELETE BOFORE MASTER VERSION*/
/*FOR BUILD.PLEASE DELETE BOFORE MASTER VERSION*/
/*FOR BUILD.PLEASE DELETE BOFORE MASTER VERSION*/
/*FOR BUILD.PLEASE DELETE BOFORE MASTER VERSION*/
cv::imwrite("/etc/urao/normal.jpg",srcImg);
/*FOR BUILD.PLEASE DELETE BOFORE MASTER VERSION*/
/*FOR BUILD.PLEASE DELETE BOFORE MASTER VERSION*/
/*FOR BUILD.PLEASE DELETE BOFORE MASTER VERSION*/
/*FOR BUILD.PLEASE DELETE BOFORE MASTER VERSION*/



resize(srcImg,small, cv::Size(), 600.0/first.cols ,337.5/first.rows);
cv::imwrite("/home/pi/Documents/php/Picture/mini.jpg",small);//For sending
	return 0;
}
