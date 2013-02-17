#include <iostream>


#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/legacy/legacy.hpp>

#include <vector>

using namespace std;
using namespace cv;



int main( int argc, char** argv )
{

    VideoCapture cap(0);
    if(!cap.isOpened())  //  Проверка корректности отработки
    {
        string message = "Camera is Broken";
        cout << message << endl;
        return -1;
    }

    Mat edges;
    namedWindow("frame",1);
    for(;;)
    {
        Mat frame, outpt;
        cap >> frame; // get a new frame from camera

        int minHessian = 400;

        SurfFeatureDetector detector( minHessian );

        std::vector<KeyPoint> keypoints_object;

        detector.detect( frame, keypoints_object );

        drawKeypoints(frame, keypoints_object, outpt, Scalar( 0, 255, 255 ), DrawMatchesFlags::DEFAULT);
        
        imshow("frame", outpt);
        if(waitKey(30) >= 0) break;
    }

}