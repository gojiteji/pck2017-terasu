#include <iostream>
#include <opencv2/opencv.hpp>
#include "opencv2/core.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/highgui.hpp"
#include <cmath>
#include <sstream>

#define HEIGHT 1080
#define WIDTH 1920

using namespace cv;
using namespace std;



int main(){	
cv::Mat first(cv::Size(WIDTH,HEIGHT), CV_8UC3);
cv::Mat result=Mat ((cv::Size(WIDTH,HEIGHT)),CV_8UC3);//最終マテリアル生成
result=cv::Scalar(0,0,0);//black画像
first =	cv::imread("/etc/urao/normal.jpg"); //<--For Demonstration cv::imread("/etc/urao/normal.jpg"); 
transpose(first,first);
cv::Mat hsv;
cvtColor(first, hsv, CV_RGB2HSV); 

for(int y=0;y<HEIGHT ;y++)
	{
	for(int x=0;x<WIDTH ;x++)
		{
		int a = hsv.step*y+(x*3);
		if( /*hsv.data[a+1] <65/*S*&&*/hsv.data[a+2] <250)//H,S,Vでの検出範囲内なら
			{
				result.data[a] = 255;
				result.data[a+1] = 255;
				result.data[a+2] = 255; 
			}
		}
	}

cv::Mat resultEx;
cv::morphologyEx(result,resultEx,cv::MORPH_OPEN,cv::Mat(),cv::Point(-1,1),1,cv::BORDER_CONSTANT,cv::morphologyDefaultBorderValue());
cv::imwrite("/etc/urao/request.jpg",resultEx);
	return 0;
}
