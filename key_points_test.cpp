#include <iostream>
#include <stdio.h>

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/legacy/legacy.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <vector>

using namespace std;
using namespace cv;



void drawVectors (const Mat& img1, 
    const vector<KeyPoint>& keypoints1, 
    const Mat& img2, 
    const vector<KeyPoint>& keypoints2, 
    const vector<DMatch>& matches1to2, 
    Mat& outImg, 
    const Scalar& matchColor=Scalar::all(-1), 
    const Scalar& singlePointColor=Scalar::all(-1), 
    const vector<char>& matchesMask=vector<char>(), 
    int flags=DrawMatchesFlags::DEFAULT)
{
    //copy image
    if( img2.type() == CV_8UC3 )
        {
            img2.copyTo( outImg );
        }
        else if( img2.type() == CV_8UC1 )
        {
            cvtColor( img2, outImg, CV_GRAY2BGR );
        }
    //It is really a fucking shit. Some constant multipliers, opencv is very interesting...
    vector<DMatch>::const_iterator match = matches1to2.begin(), last_match = matches1to2.end();
   
    //draw offset of each keypoint on second image
    while (match!=last_match)
    {
        if( matchesMask.empty())
        {
            int first_img_ind = match->queryIdx;
            int second_img_ind = match->trainIdx;
            Point center_from( keypoints1[first_img_ind].pt.x*16, keypoints1[first_img_ind].pt.y*16);
            Point center_to( keypoints2[second_img_ind].pt.x*16, keypoints2[second_img_ind].pt.y*16);
            int radius = 3*16;

            circle( outImg, center_to, radius, singlePointColor, 1, CV_AA, 4);
            line(outImg, center_from, center_to, singlePointColor, 1, CV_AA, 4);
            ++match;
        }
    }
}


int main( int argc, char** argv )
{

    VideoCapture cap(0);
    if(!cap.isOpened())  // check camera
    {
        string message = "Camera is Broken";
        cout << message << endl;
        return -1;
    }

    Mat frame_1, frame_2, outpt, outpt_kp;
    std::vector<KeyPoint> keypoints_object_1, keypoints_object_2;

    int minHessian = 300;
    SurfFeatureDetector detector( minHessian );

    namedWindow("frame",1);

    //take a snapshot from camera, as first image
    for(;;)
    {
        //show every frame with keypoints
        cap >> frame_1; // get a new frame from camera
        detector.detect( frame_1, keypoints_object_1 ); 
        drawKeypoints(frame_1, keypoints_object_1, outpt_kp, Scalar( 0, 255, 255 ), DrawMatchesFlags::DEFAULT );
        
        imshow("frame", outpt_kp);
        if(waitKey(30) >= 0) 
        {
            //save snapshot
            imwrite( "./test_img.jpg", frame_1);
            break;
        }
    }
    //and then load it as reference image
    Mat reference_image;
    reference_image = imread( "./test_img.jpg", 1 );
    detector.detect( reference_image, keypoints_object_1 );
    
    // detect keypoints offset on each frame   
    for(;;)
    {

        //detect keypoints
        cap >> frame_2; // get a new frame from camera
        detector.detect( frame_2, keypoints_object_2 );      

        SurfDescriptorExtractor extractor;
        cv::Mat descriptors1, descriptors2; 

        //kompute keypoints deskriptors
        extractor.compute(reference_image, keypoints_object_1, descriptors1);
        extractor.compute(frame_2, keypoints_object_2, descriptors2);

        //match keypoints between images
        FlannBasedMatcher matcher;
        vector< DMatch > matches;
        matcher.match(descriptors1, descriptors2, matches);

        double max_dist = 0; double min_dist = 100;

        // Quick calculation of max and min distances between keypoints
        for( int i = 0; i < descriptors1.rows; i++ )
        { 
            double dist = matches[i].distance;
            if( dist < min_dist ) min_dist = dist;
            if( dist > max_dist ) max_dist = dist;
        }

        printf("-- Max dist : %f \n", max_dist );
        printf("-- Min dist : %f \n", min_dist );

        // Draw only "good" matches (i.e. whose distance is less than some_value*min_dist )
        std::vector< DMatch > good_matches;

        for( int i = 0; i < descriptors1.rows; i++ )
        { 
            if( matches[i].distance < 3*min_dist )
            { 
                good_matches.push_back( matches[i]); 
            }
        }

        //show keypoints offset (uncomment one of these bellow)
        //drawMatches(reference_image, keypoints_object_1, frame_2, keypoints_object_2, good_matches, outpt, Scalar( 0, 255, 255 ), Scalar( 255, 0, 255 ), vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );
        drawVectors(reference_image, keypoints_object_1, frame_2, keypoints_object_2, good_matches, outpt, Scalar( 0, 255, 255 ), Scalar( 255, 0, 255 ), vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );

        imshow("frame", outpt);
        if(waitKey(30) >= 0) 
        {
            imwrite( "./test_img.jpg", frame_2);
            break;
        }
    }

}