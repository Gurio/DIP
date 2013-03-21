#include "opencv2/video/tracking.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

#include <iostream>
#include <ctype.h>

using namespace cv;
using namespace std;

static void help()
{
    // print a welcome message, and the OpenCV version
    cout << "\nThis is a demo of Lukas-Kanade optical flow lkdemo(),\n"
            "Using OpenCV version %s\n" << CV_VERSION << "\n"
            << endl;

    cout << "\nHot keys: \n"
            "\tESC - quit the program\n"
            "\tr - auto-initialize tracking\n"
            "\tc - delete all the points\n"
            "\tn - switch the \"night\" mode on/off\n"
            "To add/remove a feature point click it\n" << endl;
}

Point2f point;
bool addRemovePt = false;

static void onMouse( int event, int x, int y, int /*flags*/, void* /*param*/ )
{
    if( event == CV_EVENT_LBUTTONDOWN )
    {
        point = Point2f((float)x,(float)y);
        addRemovePt = true;
    }
}

void drawVectors (
    const vector<Point2f>& points1, 
    const Mat& img2, 
    const vector<Point2f>& points2,
    Mat& outImg, 
    const Scalar& singlePointColor=Scalar::all(-1))
{
    if( img2.type() == CV_8UC3 )
    {
        img2.copyTo( outImg );
    }
    else if( img2.type() == CV_8UC1 )
    {
        cvtColor( img2, outImg, CV_GRAY2BGR );
    }
    vector<Point2f>::const_iterator first_pt = points1.begin(), second_pt = points2.begin(), last = points1.end();
   
    //draw offset of each keypoint on second image
    while (first_pt!=last)
    {
        int radius = 3;

        circle( outImg, *second_pt, radius, singlePointColor, 1, 8);
        line(outImg, *first_pt, *second_pt, singlePointColor, 1, 8);
        ++first_pt;
        ++second_pt;
    }
}

int main( int argc, char** argv )
{
    VideoCapture cap;
    TermCriteria termcrit(CV_TERMCRIT_ITER|CV_TERMCRIT_EPS,20,0.03);
    Size subPixWinSize(10,10), winSize(31,31);

    const int MAX_COUNT = 500;
    bool needToInit = false;
    bool nightMode = false;

    if( argc == 1 || (argc == 2 && strlen(argv[1]) == 1 && isdigit(argv[1][0])))
        cap.open(argc == 2 ? argv[1][0] - '0' : 0);
    else if( argc == 2 )
        cap.open(argv[1]);

    if( !cap.isOpened() )
    {
        cout << "Could not initialize capturing...\n";
        return 0;
    }

    help();

    namedWindow( "LK Demo", 1 );
    setMouseCallback( "LK Demo", onMouse, 0 );

    Mat gray, prevGray, image, outpt, to_save;
    vector<Point2f> points[2], oldPoints;
    int framesToSkip = 10, framesSkipped = 0;

    for(;;)
    {
        Mat frame;
        cap >> frame;
        if( frame.empty() )
            break;

        frame.copyTo(image);
        cvtColor(image, gray, CV_BGR2GRAY);

        if( nightMode )
            image = Scalar::all(0);

        if( needToInit )
        {
            // automatic initialization
            goodFeaturesToTrack(gray, points[1], MAX_COUNT, 0.01, 10, Mat(), 5, 0, 0.04);
            oldPoints = points[0];
            addRemovePt = false;
        }
        else if( !points[0].empty() )
        {
            vector<uchar> status;
            vector<float> err;
            if(prevGray.empty())
                gray.copyTo(prevGray);
            calcOpticalFlowPyrLK(prevGray, gray, points[0], points[1], status, err, winSize,
                                 3, termcrit, 0, 0.001);
            size_t i, k;

        }

        if( addRemovePt && points[1].size() < (size_t)MAX_COUNT )
        {
            vector<Point2f> tmp;
            tmp.push_back(point);
            cornerSubPix( gray, tmp, winSize, cvSize(-1,-1), termcrit);
            points[1].push_back(tmp[0]);
            addRemovePt = false;
        }

        if (framesSkipped == framesToSkip)
        {
            framesSkipped = 0; 
            oldPoints = points[0]; 
            image.copyTo(to_save);
        }
        else
        {
            framesSkipped ++;
            drawVectors( oldPoints, image, points[1], outpt, Scalar( 0, 255, 255 ));
        }

        needToInit = false;
        imshow("LK Demo", outpt);

        char c = (char)waitKey(10);
        if( c == 27 )
        {
            imwrite( "./test_img_from.jpg", to_save);
            imwrite( "./test_img_to.jpg", image);
            imwrite( "./test_img_offsets.jpg", outpt);
            break;
        }
        switch( c )
        {
        case 'r':

            needToInit = true;
            break;
        case 'c':
            points[1].clear();
            break;
        case 'n':
            nightMode = !nightMode;
            break;
        default:
            ;
        }

        swap(points[1], points[0]);
        swap(prevGray, gray);
    }

    return 0;
}
