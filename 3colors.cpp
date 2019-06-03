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


int distanceX=0;
int distanceY=0;
int countX=0;
int countY=0;


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

cv::Mat TakePic(Mat Pic,string windowname="window"){
	
         cap >> Pic;
         cout<<"Pic was saved"<<endl;// break;
return Pic;
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



int main(){
SetVideo();
/*Mat current=Mat ((cv::Size(WIDTH,HEIGHT)),CV_8UC3);
for(int i=0;i<7;i++)
	current=TakePic(current,"current");
current=PinP_tr(current, current, 500, 190);
cv::Point2f center = cv::Point2f(
	static_cast<float>(current.cols ),
	static_cast<float>(current.rows ));
double degree = 0.0;  // 回転角度
double scale = 2.2;   // 拡大率(1.5倍)
//アフィン変換行列
cv::Mat affine;
cv::getRotationMatrix2D(center, degree, scale).copyTo(affine);
cv::warpAffine(current,current, affine, current.size(), cv::INTER_CUBIC);
flip(current,current, 1);
cv::imwrite("/etc/urao/current.jpg",current);
*/



     
        // FeatureDetectorオブジェクトの生成
        cv::Ptr<cv::Feature2D>  orb = cv::ORB::create();

        //画像読み込み
        cv::Mat image1 = cv::imread("/etc/urao/current.jpg");
        cv::Mat image2 = cv::imread("/etc/urao/past.jpg");

        //ORB検出
        std::vector<cv::KeyPoint> keyPoints1;
        orb->detect(image1, keyPoints1);
        std::vector<cv::KeyPoint> keyPoints2;
        orb->detect(image2, keyPoints2);

        //ORB特徴量算出
        cv::Mat descriptor1;
        cv::Mat descriptor2;
        orb->compute(image1, keyPoints1, descriptor1);
        orb->compute(image1, keyPoints2, descriptor2);
       
        //距離が近いもののみORBの対応付けをする
        static double th = 20; //pixel
        cv::BFMatcher matcher(cv::NORM_HAMMING);//cv::NORM_HAMMING);
        std::vector<cv::DMatch> matches;
       matcher.match(descriptor1, descriptor2, matches);//I added
        cv::Mat mask((keyPoints1.size()), keyPoints2.size(), CV_8U);
        mask.zeros((keyPoints1.size()), keyPoints2.size(), CV_8U);
        for (int i = 0; i < keyPoints1.size(); i++){
                cv::Point2d pt = keyPoints1[i].pt;
                for (int j = 0; j < keyPoints2.size(); j++){
                        cv::Point2d pt2 = keyPoints2[j].pt;
                        
                          cout<< pt.x<<" and   "<<pt.y<<endl;
                                distanceX+=pt.x-pt2.x;//I added
                                distanceY+=pt.y-pt2.y;//I added
                                countX++;
                                countY++;
                                
                        if (sqrt((pt.x - pt2.x)*(pt.x - pt2.x) + (pt.y - pt2.y) * (pt.y - pt2.y)) < th){
                                mask.at<uchar>(i, j) = 1;  
                        }
                }
        }
        
        

        distanceX =distanceX/countX;
        distanceY =distanceY/countY;

cout<<distanceX<<"     "<<distanceY<<endl;
  //特徴量がある程度近いもののみピックアップする
        std::vector<cv::DMatch> goodMatches;
        matcher.match(descriptor1, descriptor2, matches, mask);
        for (int i = 0; i < matches.size(); i++){
                if (matches[i].distance < 200){
                        goodMatches.push_back(matches[i]);
                }
        }

        
        //対応点表示
        cv::Mat matchImage;
        cv::drawMatches(image1, keyPoints1, image2, keyPoints2, goodMatches, matchImage, cv::Scalar::all(-1), cv::Scalar::all(-1), std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
     cv::imwrite("/etc/urao/match.jpg",matchImage);
     
     
     
     
     

 
 
 
 
 
 
 
        
        
cv::Mat first(cv::Size(WIDTH,HEIGHT), CV_8UC3);      
cv::Mat result=Mat ((cv::Size(WIDTH,HEIGHT)),CV_8UC3);//最終マテリアル生成
result=cv::Scalar(255,255,255);//黒色画像
first = cv::imread("/etc/urao/normal.jpg"); 	//for demonstration cv::imread("/etc/urao/normal.jpg"); 
transpose(first,first);//rotete 90degree
//PaintingProcess
 cv::Mat hsv,hsv2;
cvtColor(first, hsv, CV_BGR2HSV); 
for(int y=0;y<HEIGHT ;y++)
	{
	for(int x=0;x<WIDTH ;x++)
		{//hsv.rows
		int a = hsv.step*y+(x*3);
		/*black*/
		if(hsv.data[a+2]>=250 ||hsv.data[a+2] <=30 || hsv.data[a+1] <15)
			{
			result.data[a+2] = 0;
			result.data[a+1] = 0;
			result.data[a] = 0;
				}else if(hsv.data[a] <=140 && hsv.data[a] >120)
					{//blue
					result.data[a+2] = 0;
					result.data[a+1] = 0;
					result.data[a] = 255;
				}
				else if(hsv.data[a] <=50 && hsv.data[a] >10)
				{//yellow
					result.data[a+2] = 255;
					result.data[a+1] = 255;
					result.data[a] = 0;			
				}
				else if(/*hsv.data[a] <=5 ||*/ hsv.data[a] >163)
				{//red				
					result.data[a+2] = 255;
					result.data[a+1] = 0;
					result.data[a] = 0;
				}
				else if(hsv.data[a] <=163 || hsv.data[a] >140)
				{//lightred				
					result.data[a+2] = 255;
					result.data[a+1] = 100;
					result.data[a] = 200;
				}
				else if(hsv.data[a] <=120 || hsv.data[a] >50)
				{//lightgreen				
					result.data[a+2] = 0;
					result.data[a+1] = 255;
					result.data[a] = 0;
				}
		}
}
	



//average distance     
cv::Mat black=Mat ((cv::Size(WIDTH,HEIGHT)),CV_8UC3);//最終マテリアル生成
black=cv::Scalar(0,0,0);//black画像
image1=PinP_tr(black,image1, -distanceX,-distanceX);
cv::imwrite("/etc/urao/edited.jpg",image1);
 


//move redonly
cv::Mat resultEx;
black=cv::Scalar(0,0,0);//black画像

cv::morphologyEx(result,resultEx,cv::MORPH_OPEN,cv::Mat(),cv::Point(-1,1),1,cv::BORDER_CONSTANT,cv::morphologyDefaultBorderValue());
resultEx=PinP_tr(black,image1, -distanceX,-distanceX);
cout<<"complete"<<endl;
cv::imwrite("/etc/urao/answer.jpg",resultEx);
	return 0;
}
